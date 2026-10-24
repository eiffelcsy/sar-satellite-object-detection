"""Sweep the detector's inference post-processing on a trained model, without retraining.

Loads `--load` into the model built from `--config`, then scores the held-out split once per combination of the
RoI head's score threshold, NMS IoU, detections per image and image-class re-scoring (`--rescore` modes; `soft` once
per `--rescore-power`). Only inference changes: one pass per setting, no weights touched. The best setting goes into
the config as `model.score_thresh` / `nms_thresh` / `detections_per_img` / `rescore` / `rescore_power`, or on the
`--predict` command line as the matching flags, for the Codabench zips.

    python sweep_postprocess.py --config configs/ablations/g1_split_lr_200ep.yaml \
        --load runs/giou_g1_split_lr_200ep/model.pt --data "$DATA" \
        --score-thresh 0.001 --nms-thresh 0.6 --rescore off soft hard merge --rescore-power 0.5 1 2

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
    sweep.add_argument('--rescore', nargs='+', choices=['off', 'soft', 'hard', 'merge'], default=['off'],
                       help='image-class re-scoring modes to try (see sarbench/heads.py apply_class_prior)')
    sweep.add_argument('--rescore-power', nargs='+', type=float, default=[1.0], help='powers tried with soft')
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

    # soft is tried at every power; the other modes ignore the power, so they run once
    rescoring = [(mode, power) for mode in dict.fromkeys(grid.rescore)
                 for power in (grid.rescore_power if mode == 'soft' else [1.0])]
    results = []
    for score, nms, per_img, (mode, power) in itertools.product(grid.score_thresh, grid.nms_thresh,
                                                               grid.detections_per_img, rescoring):
        heads.score_thresh, heads.nms_thresh, heads.detections_per_img = score, nms, per_img
        heads.rescore, heads.rescore_power = mode, power
        metrics, _, _ = evaluate(model, loader, loader.dataset.instances_json)
        row = {'score_thresh': score, 'nms_thresh': nms, 'detections_per_img': per_img, 'rescore': mode,
               'rescore_power': power,
               **{k: metrics[k] for k in ('mAP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large', 'macro_f1')}}
        results.append(row)
        print(' '.join(f'{k} {v:.4f}' if isinstance(v, float) else f'{k} {v}' for k, v in row.items()), flush=True)

    best = max(results, key=lambda r: r['mAP'])
    print(f"best: score_thresh {best['score_thresh']} nms_thresh {best['nms_thresh']} "
          f"detections_per_img {best['detections_per_img']} rescore {best['rescore']} "
          f"rescore_power {best['rescore_power']} -> mAP {best['mAP']:.4f}")
    out = grid.sweep_out or Path(args.load).with_name('postprocess_sweep.json')
    out.write_text(json.dumps({'results': results, 'best': best}, indent=2))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
