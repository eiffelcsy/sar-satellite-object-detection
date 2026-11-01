"""Fine-tune an RF-DETR detector (Roboflow; DINOv2 backbone + DETR decoder, pretrained on COCO / Objects365)
on a dataset written by export_dataset.py.

Unlike the Faster R-CNN track, every weight is trained and the box regression starts from a detection-pretrained
model. Variants (all under the 40 M trainable budget; the script checks): small (512 px native), medium, large.
Training runs at a fixed 512 px (`multi_scale` off, so no image is ever larger than the 512 px limit) with
RF-DETR's EMA. The run writes checkpoint_best_ema.pth / checkpoint_best_total.pth and last.ckpt to --output;
`--resume` continues from last.ckpt after a time-limit kill, or extends a finished run: pass a larger --epochs
(the optimizer, EMA and epoch count come from last.ckpt; the schedule, incl. --lr-drop, from these arguments).
The LR schedule is RF-DETR's step decay: full LR, then 10x lower from epoch --lr-drop on.

Additions from the Faster R-CNN track (rfdetr_track/fusion.py; all off by default, so a plain run is unchanged):
  --dino                 fuse a frozen DINOv3 ViT-B/16 into the projector output (zero-init 1x1 convs; ~86 M frozen)
  --dino-dora-rank R     DoRA on that ViT (sarbench's), rank R, on --dino-dora-targets
  --dino-convpass-dim H  Convpass (H wide) on every ViT block
  --image-cls-weight W   an image classifier on the DINOv3 tokens, trained with weight W; predict_rfdetr.py then
                         takes the image label from it
  --p3                   add a stride-8 P3 level to the projector (P4 only by default), pretrained weights remapped
The DINOv3 settings are saved to <output>/fusion.json, which predict_rfdetr.py reads, and a --resume must match them.

Example: python rfdetr_track/train_rfdetr.py --dataset rfdetr_data/heldout_raw --output runs/rfdetr_small_raw
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fusion  # noqa: E402

VARIANTS = {'small': 'RFDETRSmall', 'medium': 'RFDETRMedium', 'large': 'RFDETRLarge', 'base': 'RFDETRBase'}
MAX_TOTAL, MAX_TRAINABLE = 130_000_000, 40_000_000


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variant', choices=sorted(VARIANTS), default='small')
    p.add_argument('--resolution', type=int, default=512, help='must be <= 512 and divisible by patch x windows')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--grad-accum', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lr-encoder', type=float, default=1.5e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--drop-path', type=float, default=0.0)
    p.add_argument('--warmup-epochs', type=float, default=1.0)
    p.add_argument('--lr-drop', type=int, default=100,
                   help='epoch from which the LR is 10x lower (RF-DETR default 100: never reached in a shorter run)')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--dino', action='store_true', help='fuse a frozen DINOv3 ViT into the projector output')
    p.add_argument('--dino-model', default=fusion.DEFAULT_DINO['model_name'])
    p.add_argument('--dino-dora-rank', type=int, default=0, help='DoRA rank on the DINOv3 ViT (0 = none)')
    p.add_argument('--dino-dora-targets', nargs='+', choices=['attn', 'mlp'], default=['attn', 'mlp'])
    p.add_argument('--dino-convpass-dim', type=int, default=0, help='Convpass width on every ViT block (0 = none)')
    p.add_argument('--image-cls-weight', type=float, default=0.0,
                   help='image-classifier loss weight (0 = no classifier; needs --dino)')
    p.add_argument('--p3', action='store_true', help='add a stride-8 P3 projector level')
    p.add_argument('--resume', action='store_true', help='continue from <output>/last.ckpt if it exists')
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--project', default='sar-satellite-object-detection')
    args = p.parse_args()
    if args.resolution > 512:
        raise SystemExit('the competition limit is 512 px')
    if not (args.dataset / 'train' / '_annotations.coco.json').exists():
        raise SystemExit(f'{args.dataset} has no train/_annotations.coco.json: run rfdetr_track/export_dataset.py '
                         f'--out {args.dataset} first (sbatchScriptRFDETR.sh step R1)')

    if (args.dino_dora_rank or args.dino_convpass_dim or args.image_cls_weight) and not args.dino:
        raise SystemExit('--dino-dora-rank / --dino-convpass-dim / --image-cls-weight need --dino')
    spec = None
    if args.dino:
        spec = fusion.full_spec({'dino': {'model_name': args.dino_model, 'dora_rank': args.dino_dora_rank,
                                          'dora_targets': args.dino_dora_targets,
                                          'convpass_dim': args.dino_convpass_dim},
                                 'image_cls_weight': args.image_cls_weight})
    saved = fusion.load_spec(args.output)
    if args.resume and (args.output / 'last.ckpt').exists() and saved != spec:
        raise SystemExit(f'--resume with different DINOv3 settings than the run in {args.output}: saved {saved}, '
                         f'now {spec}')
    fusion.install(spec)  # before any model is built; with spec None it only enables P3 weight loading

    import rfdetr
    kwargs = {'projector_scale': ['P3', 'P4']} if args.p3 else {}
    model = getattr(rfdetr, VARIANTS[args.variant])(**kwargs)  # downloads the pretrained weights on first use
    module = model.model.model
    total = sum(q.numel() for q in module.parameters())
    trainable = sum(q.numel() for q in module.parameters() if q.requires_grad)
    dino = getattr(module.backbone[0], 'dino', None)
    extra = ''
    if dino is not None:
        added = [q for n, q in module.named_parameters() if n.startswith('backbone.0.dino') or 'image_head' in n]
        extra = (f' (DINOv3 branch {sum(q.numel() for q in added):,}, of which '
                 f'{sum(q.numel() for q in added if q.requires_grad):,} trainable)')
    print(f'RF-DETR {args.variant}{" +P3" if args.p3 else ""}: {total:,} parameters, {trainable:,} trainable{extra}',
          flush=True)
    if total > MAX_TOTAL or trainable > MAX_TRAINABLE:
        raise SystemExit(f'over the parameter budget ({MAX_TOTAL:,} total / {MAX_TRAINABLE:,} trainable)')

    last = args.output / 'last.ckpt'
    resume = str(last) if args.resume and last.exists() else None
    if args.resume:
        print(f'--resume: {"continuing from " + resume if resume else "no last.ckpt yet, starting fresh"}',
              flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if spec is not None:
        fusion.save_spec(spec, args.output)
    (args.output / 'train_args.json').write_text(json.dumps(vars(args), default=str, indent=1))
    model.train(dataset_dir=str(args.dataset), output_dir=str(args.output), resolution=args.resolution,
                epochs=args.epochs, batch_size=args.batch_size, grad_accum_steps=args.grad_accum, lr=args.lr,
                lr_encoder=args.lr_encoder, weight_decay=args.weight_decay, drop_path=args.drop_path,
                warmup_epochs=args.warmup_epochs, lr_scheduler='step', lr_scheduler_kwargs={'lr_drop': args.lr_drop},
                num_workers=args.workers, multi_scale=False,
                expanded_scales=False, use_ema=True, resume=resume, wandb=args.wandb, project=args.project,
                run=args.output.name, log_per_class_metrics=True)


if __name__ == '__main__':
    main()
