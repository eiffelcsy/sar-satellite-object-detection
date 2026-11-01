"""Predict with a fine-tuned RF-DETR checkpoint: held-out metrics (same split and COCO scoring as train.py, with
per-class AP) or a Codabench submission zip for the unlabelled val / test images.

The image class (macro-F1) comes from the detections: every image holds one class, so it is the class of the
highest-scoring box. `--single-class` also relabels every box to that class (the image-class prior; off by default).
One 512 px pass per image, no test-time augmentation.

Examples:
  python rfdetr_track/predict_rfdetr.py --checkpoint runs/rfdetr_small_raw/checkpoint_best_ema.pth \\
      --data $DATA --out runs/rfdetr_small_raw/heldout
  python rfdetr_track/predict_rfdetr.py --checkpoint ... --images-json $ROOT/test/images.json --split test \\
      --out runs/rfdetr_small_raw/test
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from make_submission import write_submission  # noqa: E402
from sarbench.data import _course_train_dir, _holdout  # noqa: E402

CLASSES = ['aircraft', 'airport', 'bridge', 'car', 'harbor', 'oil_tank', 'playground', 'ship', 'wind_turbine']


def load_image(path, input_mode):
    """The model input as RF-DETR was trained on it (export_dataset.py's --input)."""
    if input_mode == 'raw':
        return Image.open(path).convert('RGB')
    from export_dataset import to_input
    return to_input(path, input_mode)


def image_predictions(detections, names, max_dets, single_class):
    """supervision Detections (original pixels) -> (image label 0-8, boxes xyxy, classes 0-8, scores), top max_dets."""
    order = np.argsort(-detections.confidence)[:max_dets]
    boxes = detections.xyxy[order]
    classes = np.array([CLASSES.index(names[int(c)]) for c in detections.class_id[order]], dtype=int)
    scores = detections.confidence[order]
    label = int(classes[0]) if len(classes) else 0
    if single_class:
        classes = np.full_like(classes, label)
    return label, boxes, classes, scores


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--data', type=Path, help='held-out evaluation: the course dataset root or its train/ dir')
    p.add_argument('--val-fraction', type=float, default=0.1)
    p.add_argument('--split-seed', type=int, default=0)
    p.add_argument('--images-json', type=Path, help='submission: an unlabelled split images.json')
    p.add_argument('--image-dir', type=Path, default=None, help='default: next to images.json, then images/')
    p.add_argument('--split', default='val', help='submission_<split>.zip')
    p.add_argument('--input', choices=['raw', 'amp_log_edge'], default='raw', help='as export_dataset.py')
    p.add_argument('--threshold', type=float, default=0.001, help='keep boxes scoring at least this')
    p.add_argument('--max-dets', type=int, default=100, help='per image, as COCO scores')
    p.add_argument('--single-class', action='store_true')
    p.add_argument('--batch', type=int, default=16)
    args = p.parse_args()
    if (args.data is None) == (args.images_json is None):
        raise SystemExit('pass exactly one of --data (held-out metrics) or --images-json (submission)')

    from rfdetr import RFDETR
    model = RFDETR.from_checkpoint(str(args.checkpoint))
    names = model.class_names
    if not set(names) <= set(CLASSES):
        raise SystemExit(f'checkpoint classes {names} are not the course classes')

    if args.data is not None:
        train_dir = _course_train_dir(args.data)
        with open(train_dir / 'labels.csv') as f:
            rows = _holdout([{'file_name': r['file_name'], 'label': int(r['label'])} for r in csv.DictReader(f)],
                            'val', args.val_fraction, args.split_seed)
        coco = json.loads((train_dir / 'instances.json').read_text())
        by_name = {Path(im['file_name']).name: im for im in coco['images']}
        images = [by_name[Path(r['file_name']).name] for r in rows]
        paths = [train_dir / 'images' / Path(im['file_name']).name for im in images]
        truth = [r['label'] for r in rows]
    else:
        images = json.loads(args.images_json.read_text())['images']
        base = args.image_dir or args.images_json.parent
        paths = []
        for im in images:
            name = Path(im['file_name']).name
            candidates = [base / im['file_name'], base / name, base / 'images' / name]
            paths.append(next((c for c in candidates if c.exists()), candidates[0]))

    sub_preds, detections, labels = {}, [], []
    for start in range(0, len(images), args.batch):
        batch = images[start:start + args.batch]
        inputs = [load_image(path, args.input) for path in paths[start:start + args.batch]]
        outputs = model.predict(inputs, threshold=args.threshold, include_source_image=False)
        outputs = outputs if isinstance(outputs, list) else [outputs]
        for im, out in zip(batch, outputs):
            label, boxes, classes, scores = image_predictions(out, names, args.max_dets, args.single_class)
            labels.append(label)
            sx, sy = 512 / im['width'], 512 / im['height']  # write_submission takes 512 px input-space boxes
            sub_preds[im['file_name']] = (label, (boxes * [sx, sy, sx, sy]).tolist(), classes.tolist(),
                                          scores.tolist())
            detections += [{'image_id': im['id'], 'category_id': int(c) + 1,
                            'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], 'score': float(s)}
                           for (x1, y1, x2, y2), c, s in zip(boxes, classes, scores)]
        print(f'{min(start + args.batch, len(images))}/{len(images)} images', flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    if args.data is not None:
        from sarbench.metrics import classification_metrics, detection_metrics
        metrics = {**classification_metrics(truth, labels),
                   **detection_metrics(train_dir / 'instances.json', detections, [im['id'] for im in images])}
        print(' '.join(f'{k} {metrics[k]:.4f}' for k in ('macro_f1', 'mAP', 'AP50', 'AP75', 'AP_small',
                                                          'AP_medium', 'AP_large')))
        print('per-class AP: ' + ' '.join(f'{k} {v:.3f}' for k, v in metrics['per_class_AP'].items()))
        (args.out / 'metrics.json').write_text(json.dumps({'args': vars(args), **metrics}, default=str, indent=1))
        (args.out / 'predictions_val.json').write_text(json.dumps(detections))
        split = 'val'
    else:
        split = args.split
    out = args.out / f'submission_{split}.zip'
    write_submission(images, sub_preds, out=out)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
