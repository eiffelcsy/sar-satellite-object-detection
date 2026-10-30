"""Find where each class's ground-truth boxes are lost on the held-out split: in the RPN or in the RoI head.

Loads `--load` into the model built from `--config`, runs the held-out split once and, for every GT box, records its
best IoU with the RPN's proposals (all of them, and the top `--top-k` by objectness) and with the final detections
(any class, and its own class with that detection's score). Prints per-class recall at each `--iou`:

    proposal       some RPN proposal reaches the GT box (the RoI head's ceiling)
    top-k          ... among the top-k proposals
    det(any)       some final detection reaches it, whatever its class
    det            a detection of its own class reaches it (what AP counts)
    lost RPN       never proposed: the fix is in the RPN (anchors, assignment, more proposals)
    lost head      proposed but not detected as its class: the fix is in the RoI head (sampling, context, loss)

Run it at the inference settings you submit with (`--score-thresh`, `--nms-thresh`, `--detections-per-img`, the usual
train.py flags); a low score threshold shows what the head ranks low rather than drops. The per-GT records go to
`recall_diagnosis.json` next to `--load` for further slicing (size, aspect ratio, image).

    python diagnose_recall.py --config configs/ablations/j1_roi_fused.yaml \\
        --load runs/giou_j1_roi_fused/model.pt --data "$DATA" --score-thresh 0.0001 --nms-thresh 0.7
"""
import argparse
import json
from pathlib import Path

import torch

from train import make_loader, parse_args
from sarbench.diagnostics import match_records, summarize
from sarbench.model import build_model


@torch.no_grad()
def main():
    diag = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    diag.add_argument('--top-k', type=int, default=300, help='also report recall among the top-k RPN proposals')
    diag.add_argument('--iou', nargs='+', type=float, default=[0.5, 0.75])
    diag.add_argument('--diag-out', type=Path, default=None, help='JSON records (default: next to --load)')
    opts, rest = diag.parse_known_args()
    args = parse_args(rest)
    if args.load is None:
        raise SystemExit('--load the trained model.pt to diagnose')
    if args.val_fraction <= 0:
        raise SystemExit('the diagnosis uses the held-out split; use val_fraction > 0')

    model, _ = build_model(args)
    model.load_state_dict(torch.load(args.load, map_location='cpu'))
    model = model.cuda().eval()
    captured = {}
    model.rpn.register_forward_hook(lambda module, inputs, output: captured.__setitem__('proposals', output[0]))
    loader = make_loader(args, 'val')
    names = {c['id']: c['name'] for c in json.loads(Path(loader.dataset.instances_json).read_text())['categories']}

    records = []
    for images, _, targets in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, outputs, _ = model(images.cuda())
        for target, proposals, out in zip(targets, captured['proposals'], outputs):
            records += match_records(target['boxes'].cuda(), target['labels'].cuda(), proposals, out['boxes'],
                                     out['labels'], out['scores'], opts.top_k, target['image_id'])

    for iou in opts.iou:
        print(f'\nrecall at IoU {iou} (score_thresh {args.score_thresh}, nms {args.nms_thresh}, '
              f'top-k {opts.top_k}):')
        print(f'{"class":13s} {"GT":>5s} {"proposal":>9s} {"top-k":>7s} {"det(any)":>9s} {"det":>6s} '
              f'{"lost RPN":>9s} {"lost head":>10s} {"med score":>10s}')
        for name, r in summarize(records, names, iou).items():
            print(f'{name:13s} {r["gt"]:5d} {r["proposal_recall"]:9.3f} {r["top_k_proposal_recall"]:7.3f} '
                  f'{r["detection_recall_any_class"]:9.3f} {r["detection_recall"]:6.3f} {r["lost_in_rpn"]:9.3f} '
                  f'{r["lost_in_roi_head"]:10.3f} {r["median_score_found"]:10.4f}')
    out = opts.diag_out or Path(args.load).with_name('recall_diagnosis.json')
    out.write_text(json.dumps({'settings': {'score_thresh': args.score_thresh, 'nms_thresh': args.nms_thresh,
                                            'detections_per_img': args.detections_per_img, 'top_k': opts.top_k},
                               'records': records}, indent=1))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
