"""The CS701 course data (https://huggingface.co/datasets/doem1997/cs701-sar-course-data).

`SARMultiTask`: the labelled train/ split (labels.csv + instances.json + images/), optionally with a deterministic
held-out part for local validation (the val/test labels are withheld). `SARImages`: an unlabelled images.json split
(val/ or test/) for prediction. Both resize every image to 512 x 512 and build the same pseudo-RGB input.
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
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.ops import box_convert
from torchvision.transforms import v2

SIZE = 512


def course_train_dir(root):
    """The train/ dir of the course data, given either the dataset root or the train/ dir itself."""
    for directory in (Path(root), Path(root) / 'train'):
        if (directory / 'labels.csv').exists() and (directory / 'instances.json').exists():
            return directory
    raise FileNotFoundError(f'no labels.csv + instances.json in {root} or {Path(root) / "train"}')


def holdout(rows, split, val_fraction, seed):
    """Split the labels.csv rows into 'train' / 'val' with a fixed seed, so both loaders agree on the split."""
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    val = set(order[:int(len(rows) * val_fraction)])
    return [row for i, row in enumerate(rows) if (i not in val) == (split == 'train')]


def _cache_name(image_path):
    """A cache-file id unique per image: the stem plus a hash of the resolved path (train/00001 and val/00001 are
    different images)."""
    digest = hashlib.md5(str(Path(image_path).resolve()).encode()).hexdigest()[:10]
    return f'{Path(image_path).stem}_{digest}'


def preprocess_image(image, image_path, preprocess, cache_dir=None):
    """The (3, H, W) pseudo-RGB tensor of one gray PIL image, reusing an on-disk .npy cache when `cache_dir` is set
    (SAR-BM3D takes ~1.5 s per image; the first epoch fills the cache)."""
    if cache_dir is not None:
        path = Path(cache_dir) / f'{_cache_name(image_path)}__{preprocess.cache_key()}.npy'
        if path.exists():
            return torch.from_numpy(np.load(path))
    arr = preprocess(np.asarray(image, dtype=np.float32) / 255.0)
    if cache_dir is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f'{path.stem}.{os.getpid()}.tmp.npy')  # must end in .npy, or np.save appends it
        np.save(tmp, arr)
        tmp.replace(path)  # atomic: concurrent workers never see a half-written file
    return torch.from_numpy(np.ascontiguousarray(arr))


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


def copy_paste(image, boxes, labels, donor_image, donor_boxes, donor_labels, max_iou=0.3, max_objects=3, margin=0):
    """Paste up to `max_objects` objects cropped from a donor image at random positions, skipping any that would
    overlap an existing box by more than `max_iou`. The crop keeps a `margin`-pixel border of context (shadow,
    smearing); the pasted box itself still hugs the object."""
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
        new_box = boxes.new_tensor([px + (x1 - cx1), py + (y1 - cy1), px + (x2 - cx1), py + (y2 - cy1)])
        if _max_iou(new_box, boxes) > max_iou:
            continue
        image[:, py:py + ch, px:px + cw] = crop
        boxes = torch.cat([boxes, new_box[None]])
        labels = torch.cat([labels, donor_labels[j][None]])
    return image, boxes, labels


def drop_degenerate_boxes(boxes, labels):
    """Drop zero-area boxes: box_iou / generalized_box_iou return NaN for them, which poisons the training loss."""
    if len(boxes) == 0:
        return boxes, labels
    keep = (boxes[:, 2] - boxes[:, 0] > 0) & (boxes[:, 3] - boxes[:, 1] > 0)
    return boxes[keep], labels[keep]


class SARMultiTask(Dataset):
    """One item is (image [3, 512, 512], label 0..8, target), target = {'boxes': XYXY on the 512 px input,
    'labels': 1..9, 'image_id': COCO id, 'orig_size': (h, w)}.

    `split` is 'train' or 'val', the two parts of train/ under `val_fraction` (0: 'train' is all of it). Training
    adds random horizontal / vertical flips and, with probability `copy_paste`, pastes up to 3 objects from another
    image of the same class (so every image keeps one class).
    """

    def __init__(self, root, split, preprocess, train=False, limit=None, val_fraction=0.0, seed=0, cache_dir=None,
                 copy_paste=0.0, copy_paste_margin=0, copy_paste_max_iou=0.3):
        train_dir = course_train_dir(root)
        self.image_dir = train_dir / 'images'
        self.instances_json = train_dir / 'instances.json'
        with open(train_dir / 'labels.csv') as f:
            rows = [{'file_name': row['file_name'], 'label': int(row['label'])} for row in csv.DictReader(f)]
        self.rows = holdout(rows, split, val_fraction, seed)[:limit]
        self.train = train
        self.copy_paste, self.copy_paste_margin, self.copy_paste_max_iou = copy_paste, copy_paste_margin, \
            copy_paste_max_iou
        self.rows_by_label = defaultdict(list)  # image label -> row indices: the copy-paste donors
        for index, row in enumerate(self.rows):
            self.rows_by_label[row['label']].append(index)
        coco = json.loads(self.instances_json.read_text())
        self.coco_images = {}  # file name (and its basename) -> COCO image record
        for img in coco['images']:
            self.coco_images[img['file_name']] = img
            self.coco_images.setdefault(Path(img['file_name']).name, img)
        self.annotations = defaultdict(list)  # COCO image id -> its boxes
        for ann in coco['annotations']:
            self.annotations[ann['image_id']].append(ann)
        flips = [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()] if train else []  # overhead: no canonical "up"
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ConvertBoundingBoxFormat('XYXY'),  # COCO [x, y, w, h] -> [x1, y1, x2, y2]
            v2.Resize((SIZE, SIZE)),  # stretches non-square images; boxes are scaled with the pixels
            *flips,
            v2.ToDtype(torch.float32, scale=True),
        ])
        self.preprocess = preprocess
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def __len__(self):
        return len(self.rows)

    def _image_path(self, file_name):
        return self.image_dir / Path(file_name).name  # labels.csv may include the images/ prefix

    def _load_transformed(self, i):
        """Image `i` after pre-processing, resize and flips: (image, XYXY boxes, labels, COCO id, original size)."""
        row = self.rows[i]
        info = self.coco_images.get(row['file_name']) or self.coco_images[Path(row['file_name']).name]
        anns = self.annotations[info['id']]
        orig_size = (info['height'], info['width'])
        boxes = tv_tensors.BoundingBoxes([a['bbox'] for a in anns], format='XYWH', canvas_size=orig_size,
                                         dtype=torch.float32)  # integer boxes would be truncated by Resize
        image = Image.open(self._image_path(row['file_name'])).convert('L')  # PIL: torchvision.io cannot read .bmp
        image = preprocess_image(image, self._image_path(row['file_name']), self.preprocess, self.cache_dir)
        image, boxes = self.transform(image, boxes)
        labels = torch.tensor([a['category_id'] for a in anns])
        return image, boxes.as_subclass(torch.Tensor).float(), labels, info['id'], orig_size

    def __getitem__(self, i):
        image, boxes, labels, image_id, orig_size = self._load_transformed(i)
        label = self.rows[i]['label']
        if self.train and self.copy_paste and random.random() < self.copy_paste:
            donor = self._load_transformed(random.choice(self.rows_by_label[label]))
            image, boxes, labels = copy_paste(image, boxes, labels, *donor[:3], self.copy_paste_max_iou,
                                              max_objects=3, margin=self.copy_paste_margin)
        if self.train:
            boxes, labels = drop_degenerate_boxes(boxes, labels)
        return image, label, {'boxes': boxes, 'labels': labels, 'image_id': image_id, 'orig_size': orig_size}


class SARImages(Dataset):
    """An unlabelled split (COCO-style images.json: id, file_name, width, height) for prediction. One item is
    (image [3, 512, 512], image_id, orig_size (h, w), file_name), built exactly like `SARMultiTask`'s input."""

    def __init__(self, images_json, preprocess, image_dir=None, cache_dir=None):
        self.images_json = Path(images_json)
        self.image_dir = Path(image_dir) if image_dir is not None else self.images_json.parent
        self.preprocess = preprocess
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.images = json.loads(self.images_json.read_text())['images']
        self.transform = v2.Compose([v2.ToImage(), v2.Resize((SIZE, SIZE)), v2.ToDtype(torch.float32, scale=True)])

    def __len__(self):
        return len(self.images)

    def _image_path(self, file_name):
        """The image on disk: the stored relative path, its basename, or images/<basename>."""
        name = Path(file_name).name
        candidates = [self.image_dir / file_name, self.image_dir / name, self.image_dir / 'images' / name]
        return next((c for c in candidates if c.exists()), candidates[0])

    def __getitem__(self, i):
        info = self.images[i]
        path = self._image_path(info['file_name'])
        image = preprocess_image(Image.open(path).convert('L'), path, self.preprocess, self.cache_dir)
        return self.transform(image), info['id'], (info['height'], info['width']), info['file_name']


def collate(batch):
    """Stack images and labels; targets stay a list (images have different numbers of boxes)."""
    images, labels, targets = zip(*batch)
    return torch.stack(images), torch.tensor(labels), list(targets)


def collate_images(batch):
    """Stack images; ids, original sizes and file names stay lists."""
    images, image_ids, orig_sizes, file_names = zip(*batch)
    return torch.stack(images), list(image_ids), list(orig_sizes), list(file_names)


def to_original_xywh(boxes_xyxy, orig_size, size=SIZE):
    """Undo the resize: XYXY boxes on the size x size input -> COCO [x, y, w, h] in original pixels."""
    h, w = orig_size
    return box_convert(boxes_xyxy * boxes_xyxy.new_tensor([w, h, w, h]) / size, 'xyxy', 'xywh')
