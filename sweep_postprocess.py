"""Sweep the detector's inference post-processing on a trained model, without retraining.

Loads `--load` into the model built from `--config`, then scores the held-out split once per combination of the
RoI head's score threshold and NMS IoU (and detections per image). Only inference changes: one pass per image, no
weights touched. The best setting goes into the config as `model.score_thresh` / `nms_thresh` /
`detections_per_img`, which `--predict` then uses for the Codabench zips.

    python sweep_postprocess.py --config configs/ablations/b4_convpass_72ep.yaml \
        --load runs/giou_b4_convpass_72ep/model.pt --data "$DATA"

COCO AP counts at most 100 detections per image, so more than 100 cannot raise mAP; the default grid keeps 100.
"""
import argparse
import itertools
import json
from pathlib import Path

import torch

from train import evaluate, make_loader, parse_args
from sarbench.model import build_model


def main():
    sweep = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sweep.add_argument('--score-thresh', nargs='+', type=float, default=[0.05, 0.01, 0.001])
    sweep.add_argument('--nms-thresh', nargs='+', type=float, default=[0.5, 0.6, 0.7])
    sweep.add_argument('--detections-per-img', nargs='+', type=int, default=[100])
    sweep.add_argument('--sweep-out', type=Path, default=None, help='JSON results (default: next to --load)')
    grid, rest = sweep.parse_known_args()
    args = parse_args(rest)  # the usual train.py flags: --config, --load, --data, --workers, ...
    if args.load is None:
        raise SystemExit('--load the trained model.pt to sweep')
    if args.val_fraction <= 0:
        raise SystemExit('the sweep scores the held-out split; use val_fraction > 0')

    model, _ = build_model(args)
    model.load_state_dict(torch.load(args.load, map_location='cpu'))
    model = model.cuda()
    loader = make_loader(args, 'val')
    heads = model.roi_heads

    results = []
    for score, nms, per_img in itertools.product(grid.score_thresh, grid.nms_thresh, grid.detections_per_img):
        heads.score_thresh, heads.nms_thresh, heads.detections_per_img = score, nms, per_img
        metrics, _, _ = evaluate(model, loader, loader.dataset.instances_json)
        row = {'score_thresh': score, 'nms_thresh': nms, 'detections_per_img': per_img,
               **{k: metrics[k] for k in ('mAP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large', 'macro_f1')}}
        results.append(row)
        print(' '.join(f'{k} {v:.4f}' if isinstance(v, float) else f'{k} {v}' for k, v in row.items()), flush=True)

    best = max(results, key=lambda r: r['mAP'])
    print(f"best: score_thresh {best['score_thresh']} nms_thresh {best['nms_thresh']} "
          f"detections_per_img {best['detections_per_img']} -> mAP {best['mAP']:.4f}")
    out = grid.sweep_out or Path(args.load).with_name('postprocess_sweep.json')
    out.write_text(json.dumps({'results': results, 'best': best}, indent=2))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
