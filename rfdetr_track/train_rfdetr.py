"""Fine-tune an RF-DETR detector (Roboflow; DINOv2 backbone + DETR decoder, pretrained on COCO / Objects365)
on a dataset written by export_dataset.py.

Unlike the Faster R-CNN track, every weight is trained and the box regression starts from a detection-pretrained
model. Variants (all under the 40 M trainable budget; the script checks): small (512 px native), medium, large.
Training runs at a fixed 512 px (`multi_scale` off, so no image is ever larger than the 512 px limit) with
RF-DETR's EMA. The run writes checkpoint_best_ema.pth / checkpoint_best_total.pth and last.ckpt to --output;
`--resume` continues from last.ckpt after a time-limit kill.

Example: python rfdetr_track/train_rfdetr.py --dataset rfdetr_data/heldout_raw --output runs/rfdetr_small_raw
"""
import argparse
from pathlib import Path

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
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--resume', action='store_true', help='continue from <output>/last.ckpt if it exists')
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--project', default='sar-satellite-object-detection')
    args = p.parse_args()
    if args.resolution > 512:
        raise SystemExit('the competition limit is 512 px')

    import rfdetr
    model = getattr(rfdetr, VARIANTS[args.variant])()  # downloads the pretrained weights on first use
    module = model.model.model
    total = sum(q.numel() for q in module.parameters())
    trainable = sum(q.numel() for q in module.parameters() if q.requires_grad)
    print(f'RF-DETR {args.variant}: {total:,} parameters, {trainable:,} trainable', flush=True)
    if total > MAX_TOTAL or trainable > MAX_TRAINABLE:
        raise SystemExit(f'over the parameter budget ({MAX_TOTAL:,} total / {MAX_TRAINABLE:,} trainable)')

    last = args.output / 'last.ckpt'
    resume = str(last) if args.resume and last.exists() else None
    if args.resume:
        print(f'--resume: {"continuing from " + resume if resume else "no last.ckpt yet, starting fresh"}',
              flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    model.train(dataset_dir=str(args.dataset), output_dir=str(args.output), resolution=args.resolution,
                epochs=args.epochs, batch_size=args.batch_size, grad_accum_steps=args.grad_accum, lr=args.lr,
                lr_encoder=args.lr_encoder, weight_decay=args.weight_decay, drop_path=args.drop_path,
                warmup_epochs=args.warmup_epochs, num_workers=args.workers, multi_scale=False,
                expanded_scales=False, use_ema=True, resume=resume, wandb=args.wandb, project=args.project,
                run=args.output.name, log_per_class_metrics=True)


if __name__ == '__main__':
    main()
