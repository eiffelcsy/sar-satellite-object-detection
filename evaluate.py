"""Score a submission zip against the released val or test labels (released 13 Nov 2026), as Codabench did.

    python evaluate.py submission_test.zip --labels data/test_labels.csv --instances data/test_instances.json
    # -> macro-F1, mAP (COCO AP at IoU 0.50:0.95) and Δm against the course reference, ViT-B/16 + LoRA

Δm = 100 x 1/2 [(F1 - F1_ref) / F1_ref + (mAP - mAP_ref) / mAP_ref], with the reference scores of the leaderboard.
"""
import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

from sarbench.metrics import classification_metrics, detection_metrics

REFERENCE = {'val': (93.31, 26.13), 'test': (93.99, 25.41)}  # ViT-B/16 + LoRA: macro-F1, mAP (percent)


def delta_m(f1, mAP, split):
    """Δm in percent for macro-F1 and mAP given in percent."""
    f1_ref, map_ref = REFERENCE[split]
    return 100 * ((f1 - f1_ref) / f1_ref + (mAP - map_ref) / map_ref) / 2


def score(submission, labels_csv, instances_json, split):
    with zipfile.ZipFile(submission) as z:
        predicted = {int(r['image_id']): int(r['label'])
                     for r in csv.DictReader(io.TextIOWrapper(z.open('classification.csv')))}
        detections = json.loads(z.read('detection.json'))
    truth = {int(r['image_id']): int(r['label']) for r in csv.DictReader(open(labels_csv))}
    if set(predicted) != set(truth):
        raise ValueError(f'{submission}: {len(predicted)} predicted images, {len(truth)} labelled ones differ')
    ids = sorted(truth)
    cls = classification_metrics([truth[i] for i in ids], [predicted[i] for i in ids])
    det = detection_metrics(str(instances_json), detections, ids)
    f1, mAP = 100 * cls['macro_f1'], 100 * det['mAP']
    return {'macro_f1': f1, 'mAP': mAP, 'AP50': 100 * det['AP50'], 'AP75': 100 * det['AP75'],
            'accuracy': 100 * cls['accuracy'], 'delta_m': delta_m(f1, mAP, split)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('submission', type=Path, help='zip written by predict.py')
    p.add_argument('--labels', type=Path, required=True, help='val_labels.csv or test_labels.csv')
    p.add_argument('--instances', type=Path, required=True, help='val_instances.json or test_instances.json')
    p.add_argument('--split', choices=REFERENCE, help='for the Δm reference (default: from the labels file name)')
    args = p.parse_args()
    split = args.split or args.labels.name.split('_')[0]
    s = score(args.submission, args.labels, args.instances, split)
    print(f"{split}: macro-F1 {s['macro_f1']:.2f}  mAP {s['mAP']:.2f}  (AP50 {s['AP50']:.2f}, AP75 {s['AP75']:.2f}, "
          f"accuracy {s['accuracy']:.2f})  delta-m {s['delta_m']:+.2f}")


if __name__ == '__main__':
    main()
