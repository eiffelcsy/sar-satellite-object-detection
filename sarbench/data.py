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


class SARMultiTask(Dataset):
    """One item is (image [C, size, size], label 0..8, target), target = {'boxes': XYXY on the
    resized image, 'labels': 1..9, 'image_id': COCO id, 'orig_size': (h, w)}; C is 1, or 3 for the
    DINOv3 pseudo-RGB pre-processing."""

    def __init__(self, root, split, train=False, size=512, limit=None, val_fraction=0.1, seed=0,
                 preprocess=None, cache_dir=None):
        self.root = Path(root)
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
        coco = json.loads(self.instances_json.read_text())
        self.coco_images = {}  # key = CSV / COCO file name, and its basename
        for img in coco['images']:
            self.coco_images[img['file_name']] = img
            self.coco_images.setdefault(Path(img['file_name']).name, img)
        self.annotations = defaultdict(list)  # COCO image id -> its boxes
        for ann in coco['annotations']:
            self.annotations[ann['image_id']].append(ann)
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

    def _image_path(self, file_name):
        if self.image_dir is None:
            return self.root / file_name
        return self.image_dir / Path(file_name).name  # labels.csv may include the images/ prefix

    def _preprocessed(self, image, file_name):
        """Despeckle/normalise the image, reusing an on-disk cache when `cache_dir` is set
        (the first pass pays for the SAR-BM3D cost, later epochs are a fast .npy load). The pre-processing
        callable may return one channel (H, W) or several (C, H, W), e.g. the DINOv3 pseudo-RGB assembly."""
        if self.cache_dir is not None:
            path = self.cache_dir / f'{Path(file_name).stem}__{self.preprocess.cache_key()}.npy'
            if path.exists():
                return _as_image_tensor(np.load(path))
        arr = self.preprocess(np.asarray(image, dtype=np.float32) / 255.0)
        if self.cache_dir is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
            np.save(tmp, arr)
            tmp.replace(path)
        return _as_image_tensor(np.ascontiguousarray(arr))

    def __getitem__(self, i):
        row = self.rows[i]
        info = self.coco_images.get(row['file_name']) or self.coco_images[Path(row['file_name']).name]
        anns = self.annotations[info['id']]
        orig_size = (info['height'], info['width'])
        boxes = tv_tensors.BoundingBoxes([a['bbox'] for a in anns], format='XYWH', canvas_size=orig_size,
                                         dtype=torch.float32)  # integer boxes would be truncated by Resize
        image = Image.open(self._image_path(row['file_name'])).convert('L')  # PIL: torchvision.io cannot read .bmp
        if self.preprocess is not None:
            image = self._preprocessed(image, row['file_name'])
        image, boxes = self.transform(image, boxes)
        target = {'boxes': boxes, 'labels': torch.tensor([a['category_id'] for a in anns]),
                  'image_id': info['id'], 'orig_size': orig_size}
        return image, row['label'], target


def collate(batch):
    """Stack images and labels; targets stay a list because images have different numbers of boxes."""
    images, labels, targets = zip(*batch)
    return torch.stack(images), torch.tensor(labels), list(targets)


def to_original_xywh(boxes_xyxy, orig_size, size=512):
    """Undo the resize: XYXY boxes on the size x size input -> COCO [x, y, w, h] in original pixels."""
    h, w = orig_size
    return box_convert(boxes_xyxy * boxes_xyxy.new_tensor([w, h, w, h]) / size, 'xyxy', 'xywh')
