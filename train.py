"""Train one model jointly for classification and detection, then score it on val and test.
Examples: python train.py --config configs/vit_lora.yaml
          python train.py --config configs/dinov3_lora.yaml"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sarbench.adapters import add_adapters
from sarbench.backbones import build_backbone
from sarbench.channels import PseudoRGB
from sarbench.config import DEFAULTS, load_config
from sarbench.data import SARMultiTask, collate, split_available, to_original_xywh
from sarbench.metrics import classification_metrics, detection_metrics
from sarbench.model import MultiTaskModel
from sarbench.preprocess import SARPreprocess


def parse_args(argv=None):
    """CLI flags override the values from --config; without --config the defaults are the reference protocol."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=Path)
    known, _ = pre.parse_known_args(argv)
    cfg = load_config(known.config) if known.config else dict(DEFAULTS)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=known.config,
                   help='YAML run config (see configs/); explicit CLI flags override its values')
    p.add_argument('--name', default=cfg['name'], help='run name (default: <backbone>_<init>_<adapt>)')
    p.add_argument('--data', type=Path, default=Path(cfg['data']) if cfg['data']
                   else Path(__file__).resolve().parents[1] / 'dataset' / 'SARFact-Course-20K')
    p.add_argument('--backbone', default=cfg['backbone'], help='backbone name from the registry (e.g. vit, terramind, dinov3)')
    p.add_argument('--init', choices=['pretrained', 'scratch'], default=cfg['init'])
    p.add_argument('--adapt', choices=['full', 'lora', 'dora', 'moelora'], default=cfg['adapt'])
    p.add_argument('--detail-stem', action='store_true', default=cfg['detail_stem'],
                   help='add a conv stem on the input to give the P2 level real stride-4 detail')
    p.add_argument('--fusion-layers', nargs='+', type=int, default=cfg['fusion_layers'],
                   help='ViT block indices whose tokens are fused before the neck (multi-layer FPN fusion)')
    p.add_argument('--head', choices=['standard', 'deform', 'cascade'], default=cfg['head'],
                   help='RoI detection head: torchvision (standard), Deformable-Conv, or Cascade R-CNN')
    p.add_argument('--epochs', type=int, default=cfg['epochs'])
    p.add_argument('--batch-size', type=int, default=cfg['batch_size'])
    p.add_argument('--lr', type=float, default=cfg['lr'])
    p.add_argument('--backbone-lr', type=float, default=cfg['backbone_lr'],
                   help='for the trainable backbone parameters (default: --lr)')
    p.add_argument('--weight-decay', type=float, default=cfg['weight_decay'])
    p.add_argument('--eval-every', type=int, default=cfg['eval_every'],
                   help='validate every N epochs, and after the last one')
    p.add_argument('--workers', type=int, default=cfg['workers'])
    p.add_argument('--limit', type=int, default=cfg['limit'],
                   help='use only the first N images of every split (quick checks)')
    p.add_argument('--val-fraction', type=float, default=cfg['val_fraction'],
                   help='course data only: fraction of train held out for validation')
    p.add_argument('--mosaic', type=float, default=cfg['mosaic'],
                   help='train-time probability of combining 4 images into a 2x2 mosaic (0 disables)')
    p.add_argument('--copy-paste', dest='copy_paste', type=float, default=cfg['copy_paste'],
                   help='train-time probability of pasting objects from another image (0 disables)')
    p.add_argument('--seed', type=int, default=cfg['seed'])
    p.add_argument('--out', type=Path, default=Path(cfg['out']))
    p.add_argument('--preprocess', action='store_true', default=cfg['preprocess'],
                   help='SAR-BM3D despeckling + dB log transform + percentile clipping before the network')
    p.add_argument('--pseudo-rgb', action='store_true', default=cfg['pseudo_rgb'],
                   help='DINOv3 input: 1-channel SAR -> 3 channels (amplitude, despeckled base, edge map)')
    p.add_argument('--edge', choices=['sobel', 'highpass'], default=cfg['edge'],
                   help='with --pseudo-rgb: the channel-3 high-frequency map (default: sobel)')
    p.add_argument('--no-despeckle', action='store_true', default=cfg['no_despeckle'],
                   help='with --preprocess/--pseudo-rgb: skip SAR-BM3D despeckling')
    p.add_argument('--no-log-transform', action='store_true', default=cfg['no_log_transform'],
                   help='with --preprocess: skip the dB transform')
    p.add_argument('--clip-percentile', type=float, default=cfg['clip_percentile'],
                   help='with --preprocess: %% of brightest pixels clipped before scaling to [0, 1]')
    p.add_argument('--preprocess-cache', type=Path,
                   default=Path(cfg['preprocess_cache']) if cfg['preprocess_cache'] else None,
                   help='with --preprocess: cache pre-processed images here (first pass fills it)')
    p.add_argument('--bm3d-sigma', type=float, default=cfg['bm3d_sigma'],
                   help='log-speckle std for SAR-BM3D (default: estimated per image)')
    p.add_argument('--bm3d-profile', default=cfg['bm3d_profile'],
                   help='bm3d package profile (np, refilter, vn, high, deb)')
    p.add_argument('--bm3d-threads', type=int, default=cfg['bm3d_threads'], help='bm3d package threads per worker')
    p.add_argument('--wandb', action='store_true', default=cfg['wandb'],
                   help='log loss/metric curves to Weights & Biases')
    p.add_argument('--wandb-project', default=cfg['wandb_project'])
    p.add_argument('--wandb-run-name', default=cfg['wandb_run_name'],
                   help='defaults to <backbone>_<init>_<adapt>')
    args = p.parse_args(argv)
    # Backbone/PEFT parameters come from the config; drop them if the CLI swapped in another backbone/method.
    args.backbone_kwargs = dict(cfg['backbone_kwargs']) if args.backbone == cfg['backbone'] else {}
    args.peft_kwargs = dict(cfg['peft_kwargs']) if args.adapt == cfg['adapt'] else {}
    args.peft_targets = list(cfg['peft_targets']) if (args.adapt == cfg['adapt'] and cfg['peft_targets']) else None
    args.head_params = dict(cfg['head_params'])
    args.loss_weights = dict(cfg['loss_weights'])
    if args.init == 'scratch' and args.adapt != 'full':
        p.error('--init scratch is only valid with --adapt full')
    if args.backbone_lr is None:
        args.backbone_lr = args.lr
    return args


def log(message, file):
    """Print to the console and to the run's log.txt."""
    print(message, flush=True)
    print(message, file=file, flush=True)


def run_name(args):
    """The config's run name, or <backbone>_<init>_<adapt> when it is absent."""
    return args.name or f'{args.backbone}_{args.init}_{args.adapt}'


def start_wandb(args, run_dir):
    """Return a live wandb run, or None when --wandb is off or wandb is not installed."""
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError:
        print('wandb not installed; skipping wandb logging (pip install wandb)', flush=True)
        return None
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    return wandb.init(project=args.wandb_project,
                      name=args.wandb_run_name or run_name(args),
                      dir=str(run_dir), config=config)


def build_preprocess(args):
    """The image pre-processing, or None when both --preprocess and --pseudo-rgb are off.

    `--pseudo-rgb` builds the three DINOv3 channels; `--preprocess` is the SAR-BM3D + dB pipeline.
    """
    if args.pseudo_rgb:
        return PseudoRGB(despeckle=not args.no_despeckle, edge=args.edge,
                         clip_percentile=args.clip_percentile, sigma=args.bm3d_sigma,
                         profile=args.bm3d_profile, threads=args.bm3d_threads)
    if not args.preprocess:
        return None
    return SARPreprocess(despeckle=not args.no_despeckle, log=not args.no_log_transform,
                         clip_percentile=args.clip_percentile, sigma=args.bm3d_sigma,
                         profile=args.bm3d_profile, threads=args.bm3d_threads)


def make_loader(args, split):
    """Batches of `split`; only the training set is shuffled and augmented."""
    train = split == 'train'
    dataset = SARMultiTask(args.data, split, train=train, limit=args.limit,
                           val_fraction=args.val_fraction, seed=args.seed,
                           preprocess=build_preprocess(args), cache_dir=args.preprocess_cache,
                           mosaic=args.mosaic, copy_paste=args.copy_paste)
    return DataLoader(dataset, args.batch_size, shuffle=train, num_workers=args.workers, collate_fn=collate,
                      persistent_workers=train and args.workers > 0)  # training workers live across epochs


def lr_factor(step, warmup, total):
    """Learning-rate multiplier: linear warmup over `warmup` steps, then cosine decay to 0 at step `total`."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))  # --epochs 1: no decay phase


def weighted_total(losses, loss_weights):
    """Sum the loss terms, scaling each by its config weight (default 1.0)."""
    weights = loss_weights or {}
    return sum(weights.get(name, 1.0) * value for name, value in losses.items())


def train_one_epoch(model, loader, optimizer, schedule, epoch, log_file, run=None, loss_weights=None):
    """One pass over the training set; the loss is cross-entropy plus the detector losses, weighted per term."""
    model.train()
    start, seen = time.time(), 0
    for step, (images, labels, targets) in enumerate(loader, 1):
        targets = [{'boxes': t['boxes'].cuda(), 'labels': t['labels'].cuda()} for t in targets]
        with torch.autocast('cuda', dtype=torch.bfloat16):  # bf16 has fp32's range: no loss scaling needed
            logits, _, det_losses = model(images.cuda(), targets)
            losses = {'classification': F.cross_entropy(logits, labels.cuda()), **det_losses}
        loss = weighted_total(losses, loss_weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        seen += len(images)
        if step % 50 == 0 or step == len(loader):
            parts = ' '.join(f'{name} {value.item():.3f}' for name, value in losses.items())
            log(f'epoch {epoch} iter {step}/{len(loader)} loss {loss.item():.3f} (weighted total; raw: {parts}) '
                f'lr {optimizer.param_groups[1]["lr"]:.2e} {seen / (time.time() - start):.1f} images/s', log_file)
            if run is not None:
                run.log({f'train/{name}': value.item() for name, value in losses.items()}
                        | {'train/loss': loss.item(), 'train/lr': optimizer.param_groups[1]['lr'],
                           'epoch': epoch},
                        step=(epoch - 1) * len(loader) + step)
            start, seen = time.time(), 0


@torch.no_grad()
def evaluate(model, loader, gt_json):
    """Classification and COCO detection metrics on one split, and the detections in COCO result format."""
    model.eval()
    labels, predicted, detections, image_ids = [], [], [], []
    for images, y, targets in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, outputs, _ = model(images.cuda())
        labels += y.tolist()
        predicted += logits.argmax(dim=1).tolist()
        for target, out in zip(targets, outputs):
            image_ids.append(target['image_id'])
            boxes = to_original_xywh(out['boxes'], target['orig_size']).tolist()
            detections += [{'image_id': target['image_id'], 'category_id': c, 'bbox': b, 'score': s}
                           for b, c, s in zip(boxes, out['labels'].tolist(), out['scores'].tolist())]
    metrics = {**classification_metrics(labels, predicted), **detection_metrics(gt_json, detections, image_ids)}
    return metrics, detections


def headline(metrics):
    return ' '.join(f'{k} {metrics[k]:.4f}' for k in ('accuracy', 'macro_f1', 'balanced_accuracy', 'mAP', 'AP50'))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    run_dir = args.out / run_name(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(run_dir / 'log.txt', 'w')
    log(json.dumps(vars(args), default=str), log_file)
    run = start_wandb(args, run_dir)

    # Data
    if (args.preprocess or args.pseudo_rgb) and args.preprocess_cache is None:
        log('--preprocess/--pseudo-rgb without --preprocess-cache: SAR-BM3D runs every epoch (slow); '
            'pass --preprocess-cache DIR to compute it once', log_file)
    train_loader, val_loader = make_loader(args, 'train'), make_loader(args, 'val')
    test_loader = make_loader(args, 'test') if split_available(args.data, 'test') else None
    if test_loader is None:
        log('no labelled test split found; skipping test evaluation', log_file)

    # Model: adapters are created on the CPU, so they are added before .cuda()
    backbone = build_backbone(args.backbone, pretrained=args.init == 'pretrained', **args.backbone_kwargs)
    if args.pseudo_rgb and backbone.in_chans != 3:
        raise SystemExit(f'--pseudo-rgb builds 3 channels but backbone {args.backbone!r} expects '
                         f'{backbone.in_chans}; use a 3-channel backbone (e.g. dinov3).')
    add_adapters(backbone, args.adapt, targets=args.peft_targets, **args.peft_kwargs)  # 'full' is a no-op
    model = MultiTaskModel(backbone, task_routing=args.adapt == 'moelora', detail_stem=args.detail_stem,
                           fusion_layers=args.fusion_layers, head=args.head,
                           head_params=args.head_params).cuda()
    parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'parameters: {parameters:,} total, {trainable:,} trainable', log_file)
    if run is not None:
        run.summary.update({'parameters': parameters, 'trainable_parameters': trainable})

    # Optimizer and schedule: trainable backbone parameters at --backbone-lr, all new modules at --lr
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for name, p in model.named_parameters() if not name.startswith('backbone.')]
    optimizer = torch.optim.AdamW([{'params': backbone_params, 'lr': args.backbone_lr}, {'params': head_params}],
                                  lr=args.lr, weight_decay=args.weight_decay)
    warmup, total = len(train_loader), args.epochs * len(train_loader)  # one epoch of warmup
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, warmup, total))

    # Train, validating every --eval-every epochs; the reported model is the last-epoch model
    history, start = [], time.time()
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, schedule, epoch, log_file, run, args.loss_weights)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val, val_detections = evaluate(model, val_loader, val_loader.dataset.instances_json)
            history.append({'epoch': epoch, **val})
            log(f'epoch {epoch} val: {headline(val)}', log_file)
            if run is not None:
                run.log({f'val/{k}': v for k, v in val.items()}, step=epoch * len(train_loader))
    train_hours = (time.time() - start) / 3600

    # Evaluate on test, once (the course val/test labels are withheld, so test may be absent)
    if test_loader is not None:
        test, test_detections = evaluate(model, test_loader, test_loader.dataset.instances_json)
        log(f'test: {headline(test)}', log_file)
    else:
        test, test_detections = None, None
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    log(f'training {train_hours:.2f} h, peak GPU memory {peak_gb:.1f} GB', log_file)
    if run is not None:
        if test is not None:
            run.log({f'test/{k}': v for k, v in test.items()})
        run.summary.update({'train_hours': train_hours, 'peak_gpu_memory_gb': peak_gb})

    # Save
    summary = {'args': vars(args), 'parameters': parameters, 'trainable_parameters': trainable,
               'train_hours': train_hours, 'peak_gpu_memory_gb': peak_gb,
               'val_history': history, 'val': history[-1], 'test': test}
    (run_dir / 'metrics.json').write_text(json.dumps(summary, indent=2, default=str))
    (run_dir / 'predictions_val.json').write_text(json.dumps(val_detections))
    if test_detections is not None:
        (run_dir / 'predictions_test.json').write_text(json.dumps(test_detections))
    torch.save(model.state_dict(), run_dir / 'model.pt')
    if run is not None:
        run.finish()


if __name__ == '__main__':
    main()
