"""Predict with a fine-tuned RF-DETR checkpoint: held-out metrics (same split and COCO scoring as train.py, with
per-class AP) or a Codabench submission zip for the unlabelled val / test images.

The image class (macro-F1) comes from the detections, since every image holds one class: the class of the
highest-scoring box (`--label-mode top`) or the class with the largest summed box score (`sum`, so one stray
confident box cannot flip it). The image class is then a prior on the boxes: `--other-class-scale s` multiplies the
scores of boxes of every other class by s (1 = off, 0 = drop them), and `--single-class` relabels every box to it.
One 512 px pass per image, no test-time augmentation.

With --data, --threshold / --label-mode / --other-class-scale take several values: the model runs once (at the
lowest threshold) and every combination is scored, ranked by the benchmark's delta-m; the best one is written.

Examples:
  python rfdetr_track/predict_rfdetr.py --checkpoint runs/rfdetr_small_raw/checkpoint_best_ema.pth \\
      --data $DATA --out runs/rfdetr_small_raw/heldout
  python rfdetr_track/predict_rfdetr.py --checkpoint ... --data $DATA --out ... \\
      --threshold 0.001 0.00001 --label-mode top sum --other-class-scale 1 0.3 0.1 0
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
F1_BASE, MAP_BASE = 93.31, 26.13  # the benchmark's delta-m reference (percent)


def delta_m(macro_f1, mAP):
    return 50 * ((100 * macro_f1 - F1_BASE) / F1_BASE + (100 * mAP - MAP_BASE) / MAP_BASE)


def load_image(path, input_mode):
    """The model input as RF-DETR was trained on it (export_dataset.py's --input)."""
    if input_mode == 'raw':
        return Image.open(path).convert('RGB')
    from export_dataset import to_input
    return to_input(path, input_mode)


def valid_detections(detections, names):
    """supervision Detections (original pixels) -> (boxes xyxy, course classes 0-8, scores) worth keeping.

    The head has len(names) + 1 outputs (the last is the no-object slot, never a training label), and RF-DETR's
    top-k runs over all of them, so a low threshold can return that slot: its boxes are dropped. So are boxes
    clipped to zero width or height at the image edge, which the submission rejects."""
    xyxy = detections.xyxy
    keep = np.flatnonzero((detections.class_id < len(names)) & (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1]))
    classes = np.array([CLASSES.index(names[int(c)]) for c in detections.class_id[keep]], dtype=int)
    return xyxy[keep], classes, detections.confidence[keep]


def image_predictions(boxes, classes, scores, threshold, label_mode, other_scale, single_class, max_dets):
    """valid_detections output -> (image label 0-8, boxes, classes, scores), the top max_dets after the prior."""
    kept = scores >= threshold
    boxes, classes, scores = boxes[kept], classes[kept], scores[kept]
    if not len(scores):
        return 0, boxes, classes, scores
    if label_mode == 'sum':
        label = int(np.argmax(np.bincount(classes, weights=scores, minlength=len(CLASSES))))
    else:
        label = int(classes[np.argmax(scores)])
    scores = np.where(classes == label, scores, scores * other_scale)
    order = np.argsort(-scores, kind='stable')[:max_dets]
    order = order[scores[order] > 0]  # other_scale 0 drops the other classes
    boxes, classes, scores = boxes[order], classes[order], scores[order]
    if single_class:
        classes = np.full_like(classes, label)
    return label, boxes, classes, scores


def collect(raw, images, setting, single_class, max_dets):
    """Apply one (threshold, label_mode, other_scale) setting to every image: labels, COCO detections, and the
    write_submission records."""
    labels, detections, sub_preds = [], [], {}
    for im, (boxes, classes, scores) in zip(images, raw):
        label, boxes, classes, scores = image_predictions(boxes, classes, scores, *setting, single_class, max_dets)
        labels.append(label)
        sx, sy = 512 / im['width'], 512 / im['height']  # write_submission takes 512 px input-space boxes
        sub_preds[im['file_name']] = (label, (boxes * [sx, sy, sx, sy]).tolist(), classes.tolist(), scores.tolist())
        detections += [{'image_id': im['id'], 'category_id': int(c) + 1,
                        'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], 'score': float(s)}
                       for (x1, y1, x2, y2), c, s in zip(boxes, classes, scores)]
    return labels, detections, sub_preds


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
    p.add_argument('--threshold', type=float, nargs='+', default=[0.001], help='keep boxes scoring at least this')
    p.add_argument('--label-mode', choices=['top', 'sum'], nargs='+', default=['top'],
                   help='image class: top-scoring box, or largest summed box score per class')
    p.add_argument('--other-class-scale', type=float, nargs='+', default=[1.0],
                   help='multiply the scores of boxes not of the image class (1 = off, 0 = drop them)')
    p.add_argument('--max-dets', type=int, default=100, help='per image, as COCO scores')
    p.add_argument('--single-class', action='store_true')
    p.add_argument('--batch', type=int, default=16)
    args = p.parse_args()
    if (args.data is None) == (args.images_json is None):
        raise SystemExit('pass exactly one of --data (held-out metrics) or --images-json (submission)')
    settings = [(t, m, o) for t in args.threshold for m in args.label_mode for o in args.other_class_scale]
    if args.images_json is not None and len(settings) > 1:
        raise SystemExit('a submission takes one --threshold / --label-mode / --other-class-scale; '
                         'sweep them with --data first')

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

    raw = []  # per image: valid boxes at the lowest threshold, post-processed per setting below
    for start in range(0, len(images), args.batch):
        inputs = [load_image(path, args.input) for path in paths[start:start + args.batch]]
        outputs = model.predict(inputs, threshold=min(args.threshold), include_source_image=False)
        outputs = outputs if isinstance(outputs, list) else [outputs]
        raw += [valid_detections(out, names) for out in outputs]
        print(f'{min(start + args.batch, len(images))}/{len(images)} images', flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    if args.data is not None:
        from sarbench.metrics import classification_metrics, detection_metrics
        image_ids = [im['id'] for im in images]
        results = []
        for setting in settings:
            labels, detections, sub_preds = collect(raw, images, setting, args.single_class, args.max_dets)
            metrics = {**classification_metrics(truth, labels),
                       **detection_metrics(train_dir / 'instances.json', detections, image_ids)}
            metrics['delta_m'] = delta_m(metrics['macro_f1'], metrics['mAP'])
            results.append((setting, metrics, detections, sub_preds))
            print(f'threshold {setting[0]:g} label {setting[1]} other-class x{setting[2]:g}: ' +
                  ' '.join(f'{k} {metrics[k]:.4f}' for k in ('delta_m', 'macro_f1', 'mAP', 'AP50', 'AP75')),
                  flush=True)
        (best, metrics, detections, sub_preds) = max(results, key=lambda r: r[1]['delta_m'])
        print(f'best: --threshold {best[0]:g} --label-mode {best[1]} --other-class-scale {best[2]:g}')
        print(' '.join(f'{k} {metrics[k]:.4f}' for k in ('delta_m', 'macro_f1', 'mAP', 'AP50', 'AP75', 'AP_small',
                                                          'AP_medium', 'AP_large')))
        print('per-class AP: ' + ' '.join(f'{k} {v:.3f}' for k, v in metrics['per_class_AP'].items()))
        sweep = [{'threshold': t, 'label_mode': m, 'other_class_scale': o,
                  **{k: v for k, v in r.items() if k != 'per_class_AP'}} for (t, m, o), r, _, _ in results]
        (args.out / 'metrics.json').write_text(json.dumps(
            {'args': vars(args), 'best': {'threshold': best[0], 'label_mode': best[1], 'other_class_scale': best[2]},
             **metrics, 'sweep': sweep}, default=str, indent=1))
        (args.out / 'predictions_val.json').write_text(json.dumps(detections))
        split = 'val'
    else:
        _, _, sub_preds = collect(raw, images, settings[0], args.single_class, args.max_dets)
        split = args.split
    out = args.out / f'submission_{split}.zip'
    write_submission(images, sub_preds, out=out)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
