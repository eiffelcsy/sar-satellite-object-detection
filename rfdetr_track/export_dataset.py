"""Write the course data as an RF-DETR dataset (Roboflow COCO layout): <out>/{train,valid,test}/ each with
_annotations.coco.json and its images.

The split is train.py's (`--val-fraction`, `--split-seed`, same defaults), so RF-DETR's held-out mAP compares with
the Faster R-CNN runs, and image / category ids are the course ones, so predictions score against instances.json.
`--input raw` links the original images (RF-DETR converts gray to RGB); `amp_log_edge` writes the 3-channel
pseudo-RGB without despeckling (amplitude, log-amplitude, Sobel edges) as PNGs. With `--val-fraction 0` (a final
model on all of train) valid/ and test/ hold a fixed 5% of train only so RF-DETR's per-epoch evaluation runs; those
numbers are then not held-out.

Example: python rfdetr_track/export_dataset.py --data $DATA --out rfdetr_data/heldout_raw
"""
import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarbench.data import _course_train_dir, _holdout  # noqa: E402


def to_input(source, input_mode):
    """The image RF-DETR trains on: None for `raw` (link the file), else a uint8 RGB PIL image."""
    if input_mode == 'raw':
        return None
    from sarbench.channels import PseudoRGB
    gray = np.asarray(Image.open(source).convert('L'), dtype=np.float32) / 255.0
    channels = PseudoRGB(mode=input_mode)(gray)  # (3, H, W) in [0, 1]
    return Image.fromarray((channels.transpose(1, 2, 0) * 255).round().astype(np.uint8))


def write_split(name, rows, coco, image_dir, out, input_mode):
    split_dir = out / name
    split_dir.mkdir(parents=True, exist_ok=True)
    by_name = {Path(im['file_name']).name: im for im in coco['images']}
    images, ids = [], set()
    for row in rows:
        info = dict(by_name[Path(row['file_name']).name])
        source = image_dir / Path(info['file_name']).name
        converted = to_input(source, input_mode)
        if converted is None:
            target = split_dir / source.name
            if not target.exists():
                try:
                    os.symlink(source.resolve(), target)
                except OSError:  # filesystems without symlinks
                    shutil.copy2(source, target)
        else:
            target = split_dir / (source.stem + '.png')
            if not target.exists():
                converted.save(target)
        info['file_name'] = target.name
        images.append(info)
        ids.add(info['id'])
    annotations = [a for a in coco['annotations'] if a['image_id'] in ids]
    (split_dir / '_annotations.coco.json').write_text(json.dumps(
        {'images': images, 'annotations': annotations, 'categories': coco['categories']}))
    print(f'{name}: {len(images)} images, {len(annotations)} boxes')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, required=True, help='the course dataset root or its train/ dir')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--val-fraction', type=float, default=0.1, help='as train.py; 0 = every image in train')
    p.add_argument('--split-seed', type=int, default=0, help='as train.py')
    p.add_argument('--input', choices=['raw', 'amp_log_edge'], default='raw')
    args = p.parse_args()

    train_dir = _course_train_dir(args.data)
    if train_dir is None:
        raise SystemExit(f'no labels.csv + instances.json under {args.data}')
    with open(train_dir / 'labels.csv') as f:
        rows = [{'file_name': r['file_name'], 'label': int(r['label'])} for r in csv.DictReader(f)]
    coco = json.loads((train_dir / 'instances.json').read_text())
    if args.val_fraction > 0:
        train_rows = _holdout(rows, 'train', args.val_fraction, args.split_seed)
        val_rows = _holdout(rows, 'val', args.val_fraction, args.split_seed)
    else:
        train_rows, val_rows = rows, _holdout(rows, 'val', 0.05, args.split_seed)  # monitoring only
    for name, split_rows in (('train', train_rows), ('valid', val_rows), ('test', val_rows)):
        write_split(name, split_rows, coco, train_dir / 'images', args.out, args.input)
    (args.out / 'export.json').write_text(json.dumps(vars(args), default=str, indent=1))


if __name__ == '__main__':
    main()
