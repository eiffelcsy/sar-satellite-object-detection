"""SAR data: every image has one class label and >= 1 box.
Loads images and boxes at a fixed input size and maps predicted boxes back to original pixels.

Two folder layouts are supported:
- the CS701 course data (root is either the dataset dir or its train/ dir): labels.csv +
  instances.json + images/. train and val are a deterministic split of labels.csv, because the
  course val/test labels are withheld;
- the original reference layout: classification/{split}.csv + detection/instances_{split}.json,
  with image paths relative to root.
"""
import csv
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import tv_tensors
from torchvision.ops import box_convert
from torchvision.transforms import v2

# Mean and std of all pixels of the 9,392 train images, read as gray, resized to
# 512 x 512 and scaled to [0, 1] (computed once).
MEAN, STD = 0.2526, 0.2133


def _course_train_dir(root):
    """The train/ dir of the course layout, given either the dataset root or the train/ dir itself."""
    for directory in (Path(root), Path(root) / 'train'):
        if (directory / 'labels.csv').exists() and (directory / 'instances.json').exists():
            return directory
    return None


def _holdout(rows, split, val_fraction, seed):
    """Split labels.csv rows into train/val with a fixed seed, so both loaders agree on the split."""
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    n_val = int(len(rows) * val_fraction)
    val = set(order[:n_val])
    return [row for i, row in enumerate(rows) if (i not in val) == (split == 'train')]


def split_available(root, split):
    """Whether `split` can be built from `root` in either layout (val/test labels may be missing)."""
    if split in ('train', 'val') and _course_train_dir(root) is not None:
        return True
    return (Path(root) / 'classification' / f'{split}.csv').exists() \
        and (Path(root) / 'detection' / f'instances_{split}.json').exists()


def _as_image_tensor(arr):
    """A pre-processed (H, W) or (C, H, W) array -> a (1, H, W) or (C, H, W) float32 image tensor."""
    tensor = torch.from_numpy(np.ascontiguousarray(arr).astype(np.float32, copy=False))
    return tensor.unsqueeze(0) if tensor.ndim == 2 else tensor


def _cache_name(image_path):
    """A cache-file id unique per image: the stem plus a hash of the resolved path.

    The path hash keeps images with the same stem in different splits (e.g. train/00001 and val/00001)
    from colliding in one cache directory.
    """
    digest = hashlib.md5(str(Path(image_path).resolve()).encode()).hexdigest()[:10]
    return f'{Path(image_path).stem}_{digest}'


def preprocess_image(image, image_path, preprocess, cache_dir=None):
    """Despeckle/normalise one PIL image, reusing an on-disk cache when `cache_dir` is set.

    The first pass pays for the SAR-BM3D cost, later runs are a fast .npy load. The pre-processing callable
    may return one channel (H, W) or several (C, H, W), e.g. the DINOv3 pseudo-RGB assembly. Shared by the
    labelled and the images-only datasets so both build the identical network input.
    """
    if cache_dir is not None:
        path = Path(cache_dir) / f'{_cache_name(image_path)}__{preprocess.cache_key()}.npy'
        if path.exists():
            return _as_image_tensor(np.load(path))
    arr = preprocess(np.asarray(image, dtype=np.float32) / 255.0)
    if cache_dir is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # The temp name must end in .npy: np.save appends .npy to any other suffix, so the replace below
        # would otherwise look for a file that was never written. The pid keeps concurrent workers apart.
        tmp = path.with_name(f'{path.stem}.{os.getpid()}.tmp.npy')
        np.save(tmp, arr)
        tmp.replace(path)  # atomic: a half-written cache file is never visible to another worker
    return _as_image_tensor(np.ascontiguousarray(arr))


def _max_iou(box, boxes):
    """Largest IoU of one XYXY box against a set of XYXY boxes."""
    if len(boxes) == 0:
        return 0.0
    lt = torch.maximum(box[None, :2], boxes[:, :2])
    rb = torch.minimum(box[None, 2:], boxes[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area = (box[2] - box[0]) * (box[3] - box[1])
    union = area + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter
    return (inter / union.clamp(min=1e-6)).max().item()


def mosaic_tiles(tiles, size):
    """Combine four (image [C,size,size], boxes XYXY, labels) into one size x size 2x2 mosaic.

    Each tile is resized to size/2 and placed in a quadrant; boxes are scaled and clipped to their quadrant.
    Images are already on the input canvas (512 px), so the coordinates stay consistent with the other paths.
    """
    half = size // 2
    canvas = torch.zeros_like(tiles[0][0])
    boxes_out, labels_out = [], []
    for (image, boxes, labels), (ox, oy) in zip(tiles, [(0, 0), (half, 0), (0, half), (half, half)]):
        canvas[:, oy:oy + half, ox:ox + half] = F.interpolate(
            image.unsqueeze(0), size=(half, half), mode='bilinear', align_corners=False)[0]
        if len(boxes):
            scaled = boxes * (half / size) + boxes.new_tensor([ox, oy, ox, oy])
            scaled[:, 0::2].clamp_(ox, ox + half)
            scaled[:, 1::2].clamp_(oy, oy + half)
            keep = (scaled[:, 2] - scaled[:, 0] > 1) & (scaled[:, 3] - scaled[:, 1] > 1)
            boxes_out.append(scaled[keep])
            labels_out.append(labels[keep])
    boxes = torch.cat(boxes_out) if boxes_out else tiles[0][1][:0]
    labels = torch.cat(labels_out) if labels_out else tiles[0][2][:0]
    return canvas, boxes, labels


def _inside_fraction(boxes, left, top, size):
    """Fraction of each box's area inside the `size` x `size` window whose top-left corner is (left, top)."""
    x1 = boxes[:, 0].clamp(min=left, max=left + size)
    y1 = boxes[:, 1].clamp(min=top, max=top + size)
    x2 = boxes[:, 2].clamp(min=left, max=left + size)
    y2 = boxes[:, 3].clamp(min=top, max=top + size)
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / area.clamp(min=1e-6)


def scale_jitter(image, boxes, labels, scale_range=(0.9, 1.1), size=512, min_inside=0.5):
    """Resize by a scale near 1.0: shrinking pads the border (nothing is lost), enlarging crops.

    On enlargement the crop window is the candidate that keeps the most boxes; a box left less than
    `min_inside` of its area in the window is dropped rather than truncated.
    """
    scale = random.uniform(*scale_range)
    extent = max(1, int(round(size * scale)))
    if extent == size:
        return image, boxes, labels
    resized = F.interpolate(image.unsqueeze(0), size=(extent, extent), mode='bilinear', align_corners=False)[0]
    boxes = boxes * scale
    if extent < size:  # shrink: pad back to `size`, keeping every box and pixel
        pad = (size - extent) // 2
        canvas = image.new_zeros(image.shape)
        canvas[:, pad:pad + extent, pad:pad + extent] = resized
        return canvas, boxes + boxes.new_tensor([pad, pad, pad, pad]), labels
    offsets = {(0, 0), (extent - size, extent - size)}
    if len(boxes):  # candidate windows anchored on the boxes' centre and corners
        cx = ((boxes[:, 0] + boxes[:, 2]) / 2).mean().item()
        cy = ((boxes[:, 1] + boxes[:, 3]) / 2).mean().item()
        offsets.add((int(cx - size / 2), int(cy - size / 2)))
        for x1, y1, x2, y2 in boxes.tolist():
            offsets |= {(int(x1), int(y1)), (int(x2) - size, int(y2) - size)}
    candidates = [(min(max(x, 0), extent - size), min(max(y, 0), extent - size)) for x, y in offsets]
    left, top = max(candidates, key=lambda o: _inside_fraction(boxes, o[0], o[1], size).ge(min_inside).sum())
    keep = _inside_fraction(boxes, left, top, size) >= min_inside
    cropped = resized[:, top:top + size, left:left + size]
    moved = (boxes[keep] - boxes.new_tensor([left, top, left, top])).clamp(min=0, max=size)
    return cropped, moved, labels[keep]


def rotate90(image, boxes, k):
    """Rotate a square [C, S, S] image and its XYXY boxes by k x 90 degrees (counter-clockwise), exactly.

    torch.rot90 moves pixel (x, y) to (y, S - 1 - x), so a box edge at x maps to S - x: no interpolation and no
    sub-pixel drift, unlike an affine rotation about the image centre.
    """
    size = image.shape[-1]
    for _ in range(k % 4):
        image = torch.rot90(image, 1, dims=(1, 2))
        boxes = torch.stack([boxes[:, 1], size - boxes[:, 2], boxes[:, 3], size - boxes[:, 0]], dim=1)
    return image, boxes


def drop_degenerate_boxes(boxes, labels):
    """Drop zero-area boxes: `box_iou`/`generalized_box_iou` return NaN for them, which poisons the training loss.

    A geometric augmentation (crop or resize) can clip a box down to a line; such boxes carry no information and
    must not reach the matcher or the GIoU term.
    """
    if len(boxes) == 0:
        return boxes, labels
    keep = (boxes[:, 2] - boxes[:, 0] > 0) & (boxes[:, 3] - boxes[:, 1] > 0)
    return boxes[keep], labels[keep]


def copy_paste(image, boxes, labels, donor_image, donor_boxes, donor_labels, max_iou=0.3, max_objects=3,
               margin=0):
    """Paste up to `max_objects` objects cropped from a donor image, avoiding overlaps with existing boxes.

    The crop takes a `margin`-pixel border around the box, so the object's SAR signature (shadow, smearing,
    wakes) is not cut off; the pasted box itself still hugs the object.
    """
    _, height, width = image.shape
    for j in torch.randperm(len(donor_boxes))[:max_objects]:
        x1, y1, x2, y2 = donor_boxes[j].round().long().tolist()
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        cx1, cy1 = max(0, x1 - margin), max(0, y1 - margin)
        cx2, cy2 = min(width, x2 + margin), min(height, y2 + margin)
        crop = donor_image[:, cy1:cy2, cx1:cx2]
        ch, cw = crop.shape[1], crop.shape[2]
        px = int(torch.randint(0, width - cw + 1, (1,))) if width > cw else 0
        py = int(torch.randint(0, height - ch + 1, (1,))) if height > ch else 0
        new_box = boxes.new_tensor([px + (x1 - cx1), py + (y1 - cy1),
                                    px + (x2 - cx1), py + (y2 - cy1)])
        if _max_iou(new_box, boxes) > max_iou:
            continue
        image[:, py:py + ch, px:px + cw] = crop
        boxes = torch.cat([boxes, new_box[None]])
        labels = torch.cat([labels, donor_labels[j][None]])
    return image, boxes, labels


class SARMultiTask(Dataset):
    """One item is (image [C, size, size], label 0..8, target), target = {'boxes': XYXY on the
    resized image, 'labels': 1..9, 'image_id': COCO id, 'orig_size': (h, w)}; C is 1, or 3 for the
    DINOv3 pseudo-RGB pre-processing."""

    def __init__(self, root, split, train=False, size=512, limit=None, val_fraction=0.1, seed=0,
                 preprocess=None, cache_dir=None, mosaic=0.0, copy_paste=0.0, copy_paste_max_iou=0.3,
                 copy_paste_margin=0, copy_paste_same_class=False, copy_paste_small=None, scale_jitter=None,
                 rotation90=False):
        self.root = Path(root)
        self.train = train
        self.size = size
        self.mosaic = mosaic  # probability of building a 2x2 mosaic from four images (train only)
        self.copy_paste = copy_paste  # probability of pasting objects from another image (train only)
        self.copy_paste_max_iou = copy_paste_max_iou
        self.copy_paste_margin = copy_paste_margin  # px of context pasted around each object
        # Donors from images of the same class only: every image keeps one class (its label stays true) and a pasted
        # object lands in a scene of its own kind (a ship in a harbour scene, not an airport).
        self.copy_paste_same_class = copy_paste_same_class
        # Stronger copy-paste for images of small-object classes (most of their objects COCO-small, area < 32^2 in
        # original pixels; with this dataset: bridge, oil tank, playground, wind turbine, ship): with probability
        # `prob` instead of `copy_paste`, from up to `donors` donor images, until `max_objects` objects are pasted.
        # The other classes keep `copy_paste`. None disables it. `min_small_fraction` (0.5) picks the classes.
        self.copy_paste_small = dict(copy_paste_small) if copy_paste_small else None
        self.small_object_labels = set()
        self.rotation90 = rotation90  # random 0/90/180/270-degree rotation (train only)
        self.scale_jitter = scale_jitter  # (lo, hi) scale range applied after the resize, or None/False
        self.image_dir = None
        train_dir = _course_train_dir(self.root) if split in ('train', 'val') else None
        if train_dir is not None:  # course data: hold out part of train for local validation
            self.image_dir = train_dir / 'images'
            self.instances_json = train_dir / 'instances.json'
            with open(train_dir / 'labels.csv') as f:
                rows = [{'file_name': row['file_name'], 'label': int(row['label'])}
                        for row in csv.DictReader(f)]
            rows = _holdout(rows, split, val_fraction, seed)
        else:  # reference layout: one labelled file per split
            self.instances_json = self.root / 'detection' / f'instances_{split}.json'
            with open(self.root / 'classification' / f'{split}.csv') as f:
                rows = [{'file_name': row['image'], 'label': int(row['label'])}
                        for row in csv.DictReader(f)]
        self.rows = rows[:limit]
        self.rows_by_label = defaultdict(list)  # image label -> row indices (same-class copy-paste donors)
        for index, row in enumerate(self.rows):
            self.rows_by_label[row['label']].append(index)
        coco = json.loads(self.instances_json.read_text())
        self.coco_images = {}  # key = CSV / COCO file name, and its basename
        for img in coco['images']:
            self.coco_images[img['file_name']] = img
            self.coco_images.setdefault(Path(img['file_name']).name, img)
        self.annotations = defaultdict(list)  # COCO image id -> its boxes
        for ann in coco['annotations']:
            self.annotations[ann['image_id']].append(ann)
        if self.copy_paste_small:
            self.small_object_labels = self._small_object_labels(self.copy_paste_small.get('min_small_fraction', 0.5))
        flips = [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()] if train else []
        steps = [
            v2.ToImage(),
            v2.ConvertBoundingBoxFormat('XYXY'),  # COCO [x, y, w, h] -> [x1, y1, x2, y2]
            v2.Resize((size, size)),  # stretches non-square images; boxes are scaled with the pixels
            *flips,  # overhead imagery has no canonical "up"
            v2.ToDtype(torch.float32, scale=True),
        ]
        # Pre-processing already returns a [0, 1] image, so the raw-pixel MEAN/STD no longer apply.
        if preprocess is None:
            steps.append(v2.Normalize([MEAN], [STD]))
        self.transform = v2.Compose(steps)
        self.preprocess = preprocess
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def __len__(self):
        return len(self.rows)

    def sampling_weights(self, weights_by_name):
        """One weight per row: `weights_by_name[class name]` for images of that class, 1 otherwise."""
        names = {c['id'] - 1: c['name'] for c in json.loads(self.instances_json.read_text())['categories']}
        unknown = set(weights_by_name) - set(names.values())
        if unknown:
            raise ValueError(f"unknown class in class_sampling_weights: {sorted(unknown)}; "
                             f"available: {', '.join(names.values())}")
        return [float(weights_by_name.get(names[row['label']], 1.0)) for row in self.rows]

    def _small_object_labels(self, min_fraction):
        """Image labels whose objects (over these rows) are at least `min_fraction` COCO-small (area < 32^2 px)."""
        small, total = defaultdict(int), defaultdict(int)
        for row in self.rows:
            info = self.coco_images.get(row['file_name']) or self.coco_images[Path(row['file_name']).name]
            for ann in self.annotations[info['id']]:
                total[row['label']] += 1
                small[row['label']] += ann['bbox'][2] * ann['bbox'][3] < 32 ** 2
        return {label for label in total if small[label] >= min_fraction * total[label]}

    def _image_path(self, file_name):
        if self.image_dir is None:
            return self.root / file_name
        return self.image_dir / Path(file_name).name  # labels.csv may include the images/ prefix

    def _preprocessed(self, image, file_name):
        """Despeckle/normalise the image, reusing an on-disk cache when `cache_dir` is set."""
        return preprocess_image(image, self._image_path(file_name), self.preprocess, self.cache_dir)

    def _load_transformed(self, i):
        """Load image `i`, run the base transform (resize/flips/normalise) and return image, boxes, labels.

        `boxes` are plain XYXY float tensors on the `size x size` input canvas, ready for the augmentation
        helpers; image_id / orig_size are kept so predicted boxes can be mapped back for COCO scoring.
        """
        row = self.rows[i]
        info = self.coco_images.get(row['file_name']) or self.coco_images[Path(row['file_name']).name]
        anns = self.annotations[info['id']]
        orig_size = (info['height'], info['width'])
        boxes = tv_tensors.BoundingBoxes([a['bbox'] for a in anns], format='XYWH', canvas_size=orig_size,
                                         dtype=torch.float32)  # integer boxes would be truncated by Resize
        image = Image.open(self._image_path(row['file_name'])).convert('L')  # PIL: torchvision.io cannot read .bmp
        if self.preprocess is not None:  # filter channels are computed (and cached) before any augmentation
            image = self._preprocessed(image, row['file_name'])
        image, boxes = self.transform(image, boxes)
        boxes = boxes.as_subclass(torch.Tensor).float()
        labels = torch.tensor([a['category_id'] for a in anns])
        if self.train and self.rotation90:  # 0/90/180/270, free for SAR (flips already break the look direction)
            image, boxes = rotate90(image, boxes, random.randrange(4))
        if self.train and self.scale_jitter:  # gentle, centred on 1.0; pads when shrinking, crops when enlarging
            image, boxes, labels = scale_jitter(image, boxes, labels, tuple(self.scale_jitter), self.size)
        return image, boxes, labels, info['id'], orig_size

    def _paste(self, image, boxes, labels, label, donors=1, max_objects=3):
        """Copy-paste from up to `donors` donor images (of `label` with copy_paste_same_class), `max_objects` in all."""
        pool = self.rows_by_label[label] if self.copy_paste_same_class else range(len(self))
        start = len(boxes)
        for _ in range(donors):
            remaining = max_objects - (len(boxes) - start)
            if remaining <= 0:
                break
            donor = self._load_transformed(random.choice(pool))
            image, boxes, labels = copy_paste(image, boxes, labels, donor[0], donor[1], donor[2],
                                              self.copy_paste_max_iou, max_objects=remaining,
                                              margin=self.copy_paste_margin)
        return image, boxes, labels

    def __getitem__(self, i):
        image, boxes, labels, image_id, orig_size = self._load_transformed(i)
        if self.train and self.mosaic and random.random() < self.mosaic:  # 4 images -> one 2x2 mosaic
            tiles = [(image, boxes, labels)] + [self._load_transformed(random.randrange(len(self)))[:3]
                                                for _ in range(3)]
            image, boxes, labels = mosaic_tiles(tiles, self.size)
        label = self.rows[i]['label']
        if self.train and self.copy_paste_small and label in self.small_object_labels:
            if random.random() < self.copy_paste_small.get('prob', 1.0):  # several donors, up to max_objects
                image, boxes, labels = self._paste(image, boxes, labels, label,
                                                   self.copy_paste_small.get('donors', 4),
                                                   self.copy_paste_small.get('max_objects', 8))
        elif self.train and self.copy_paste and random.random() < self.copy_paste:  # paste donor objects
            image, boxes, labels = self._paste(image, boxes, labels, label)
        if self.train:  # zero-area boxes (from the crop) make GIoU/box_iou NaN, so drop them before the model
            boxes, labels = drop_degenerate_boxes(boxes, labels)
        target = {'boxes': boxes, 'labels': labels, 'image_id': image_id, 'orig_size': orig_size}
        return image, self.rows[i]['label'], target


class RepeatFactorSampler(Sampler):
    """Every image once per epoch plus repeats: image i appears floor(w_i) times, and the fractional parts add
    round(sum of fractions) more draws (weighted, without replacement), so each epoch has the same length.

    Unlike WeightedRandomSampler, no image is ever skipped; the epoch grows instead (LVIS repeat-factor sampling).
    """

    def __init__(self, weights):
        weights = torch.as_tensor(weights, dtype=torch.float64)
        if (weights < 0).any():
            raise ValueError('repeat factors must be >= 0')
        self.whole = weights.floor().long()
        self.frac = weights - self.whole
        self.extra = int(round(self.frac.sum().item()))
        self.length = int(self.whole.sum().item()) + self.extra

    def __len__(self):
        return self.length

    def __iter__(self):
        indices = torch.repeat_interleave(torch.arange(len(self.whole)), self.whole)
        if self.extra:
            indices = torch.cat([indices, torch.multinomial(self.frac, self.extra, replacement=False)])
        yield from indices[torch.randperm(len(indices))].tolist()


def collate(batch):
    """Stack images and labels; targets stay a list because images have different numbers of boxes."""
    images, labels, targets = zip(*batch)
    return torch.stack(images), torch.tensor(labels), list(targets)


class SARImages(Dataset):
    """An images-only split (COCO `images.json`, no annotations or labels) for submission prediction.

    One item is (image [C, size, size], image_id, orig_size (h, w), file_name). The input pipeline is the same
    as `SARMultiTask` (resize to `size`, normalise, optional SAR pre-processing/cache); only the boxes and the
    class label are absent, because the unlabelled val/test splits have none and the submission only needs the
    model's predictions. `file_name` is the key used to match a prediction to its image.
    """

    def __init__(self, images_json, image_dir=None, size=512, preprocess=None, cache_dir=None):
        self.images_json = Path(images_json)
        self.image_dir = Path(image_dir) if image_dir is not None else self.images_json.parent
        self.size = size
        self.preprocess = preprocess
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.images = json.loads(self.images_json.read_text())['images']
        steps = [v2.ToImage(), v2.Resize((size, size)), v2.ToDtype(torch.float32, scale=True)]
        if preprocess is None:  # pre-processing already returns a [0, 1] image, so MEAN/STD do not apply
            steps.append(v2.Normalize([MEAN], [STD]))
        self.transform = v2.Compose(steps)

    def __len__(self):
        return len(self.images)

    def _image_path(self, file_name):
        """The image on disk: try the stored relative path, its basename, then an images/ subdir."""
        name = Path(file_name).name
        candidates = [self.image_dir / file_name, self.image_dir / name, self.image_dir / 'images' / name]
        return next((c for c in candidates if c.exists()), candidates[0])

    def __getitem__(self, i):
        info = self.images[i]
        file_name = info['file_name']
        image = Image.open(self._image_path(file_name)).convert('L')  # PIL: torchvision.io cannot read .bmp
        if self.preprocess is not None:
            image = preprocess_image(image, self._image_path(file_name), self.preprocess, self.cache_dir)
        image = self.transform(image)
        return image, info['id'], (info['height'], info['width']), file_name


def collate_images(batch):
    """Stack images; ids, original sizes and file names stay lists (they identify each image)."""
    images, image_ids, orig_sizes, file_names = zip(*batch)
    return torch.stack(images), list(image_ids), list(orig_sizes), list(file_names)


def to_original_xywh(boxes_xyxy, orig_size, size=512):
    """Undo the resize: XYXY boxes on the size x size input -> COCO [x, y, w, h] in original pixels."""
    h, w = orig_size
    return box_convert(boxes_xyxy * boxes_xyxy.new_tensor([w, h, w, h]) / size, 'xyxy', 'xywh')
