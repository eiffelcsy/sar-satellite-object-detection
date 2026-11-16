"""Train the final model (configs/final.yaml) jointly for image classification and object detection.

    python train.py --data /path/to/cs701-sar-course-data/train            # the final model: 250 epochs, all of train
    python train.py --data ... --val-fraction 0.1 --epochs 96             # with a held-out split, validated every 4
    python train.py --data ... --epochs 1 --limit 64                      # smoke test

Writes runs/<name>/: log.txt, metrics.json and model.pt (the full state_dict, the input of predict.py). After every
epoch it writes last.pt; if the job is cut off, rerun the same command with --resume to continue it.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sarbench.checkpoint import load_checkpoint, save_checkpoint
from sarbench.config import load_config
from sarbench.data import SARMultiTask, collate, to_original_xywh
from sarbench.metrics import classification_metrics, detection_metrics
from sarbench.model import MAX_TOTAL_PARAMETERS, MAX_TRAINABLE_PARAMETERS, build_model, parameter_counts
from sarbench.preprocess import PseudoRGB

ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=ROOT / 'configs' / 'final.yaml')
    p.add_argument('--data', type=Path, required=True, help='the course data train/ dir (or the dataset root)')
    p.add_argument('--out', type=Path, default=Path('runs'), help='runs are written to <out>/<config name>/')
    p.add_argument('--resume', action='store_true', help='continue from <run dir>/last.pt if it exists')
    p.add_argument('--wandb', action='store_true', help='log losses and metrics to Weights & Biases')
    p.add_argument('--wandb-project', default='sar-satellite-object-detection')
    # overrides of the config, for quick checks (the final model uses none of them)
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--workers', type=int)
    p.add_argument('--seed', type=int)
    p.add_argument('--val-fraction', type=float, help='hold out this fraction of train for validation')
    p.add_argument('--limit', type=int, help='use only the first N images (smoke tests)')
    args = p.parse_args(argv)
    config = load_config(args.config)
    for key in ('epochs', 'batch_size', 'workers', 'seed'):
        if getattr(args, key) is not None:
            config['train'][key] = getattr(args, key)
    if args.val_fraction is not None:
        config['data']['val_fraction'] = args.val_fraction
    return args, config


def log(message, file):
    """Print to the console and to the run's log.txt."""
    print(message, flush=True)
    print(message, file=file, flush=True)


def make_loader(args, config, split):
    """Batches of `split` ('train' or the held-out 'val'); only the training set is shuffled and augmented."""
    train = split == 'train'
    data, cfg = config['data'], config['train']
    dataset = SARMultiTask(args.data, split, PseudoRGB(config['preprocess']['clip_percentile']), train=train,
                           limit=args.limit, val_fraction=data['val_fraction'], seed=data['split_seed'],
                           cache_dir=config['preprocess']['cache'], copy_paste=data['copy_paste'],
                           copy_paste_margin=data['copy_paste_margin'])
    return DataLoader(dataset, cfg['batch_size'], shuffle=train, num_workers=cfg['workers'], collate_fn=collate,
                      persistent_workers=train and cfg['workers'] > 0)  # training workers live across epochs


def lr_factor(step, warmup, total):
    """Learning-rate multiplier: linear warmup over `warmup` steps, then cosine decay to 0 at step `total`."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))


def weighted_total(losses, loss_weights):
    """Sum the loss terms, scaling each by its config weight (default 1.0)."""
    return sum(loss_weights.get(name, 1.0) * value for name, value in losses.items())


def train_one_epoch(model, loader, optimizer, schedule, epoch, loss_weights, log_file, run=None):
    """One pass over the training set: cross-entropy for the image class plus the detector losses, weighted."""
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
                f'lr {optimizer.param_groups[-1]["lr"]:.2e} {seen / (time.time() - start):.1f} images/s', log_file)
            if run is not None:
                run.log({f'train/{name}': value.item() for name, value in losses.items()}
                        | {'train/loss': loss.item(), 'train/lr': optimizer.param_groups[-1]['lr'], 'epoch': epoch},
                        step=(epoch - 1) * len(loader) + step)
            start, seen = time.time(), 0


@torch.no_grad()
def evaluate(model, loader):
    """Macro-F1 / accuracy and COCO box AP on the labelled held-out split."""
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
    return {**classification_metrics(labels, predicted),
            **detection_metrics(loader.dataset.instances_json, detections, image_ids)}


def headline(metrics):
    return ' '.join(f'{k} {metrics[k]:.4f}' for k in ('accuracy', 'macro_f1', 'balanced_accuracy', 'mAP', 'AP50',
                                                      'AP75', 'AP_small', 'AP_medium', 'AP_large'))


def main():
    args, config = parse_args()
    cfg = config['train']
    torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for the fp32 RPN / RoI head, as in training the final model
    torch.backends.cudnn.allow_tf32 = True
    run_dir = args.out / config['name']
    run_dir.mkdir(parents=True, exist_ok=True)
    last = run_dir / 'last.pt'
    resume = args.resume and last.exists()
    log_file = open(run_dir / 'log.txt', 'a' if resume else 'w')  # a resumed run continues the same log
    log(json.dumps({'args': vars(args), 'config': config}, default=str), log_file)
    run = None
    if args.wandb:  # a resumed run continues the same wandb run: its id is in the checkpoint
        import wandb
        run_id = torch.load(last, map_location='cpu', weights_only=False)['extra'].get('wandb_id') if resume else None
        run = wandb.init(project=args.wandb_project, name=config['name'], dir=str(run_dir), config=config, id=run_id,
                         resume='allow' if run_id else None)

    train_loader = make_loader(args, config, 'train')
    val_loader = make_loader(args, config, 'val') if config['data']['val_fraction'] > 0 else None
    if val_loader is None:
        log('val_fraction 0: training on all of train, no held-out validation', log_file)

    # Model: the adapters are created on the CPU, before .cuda()
    model, pretrained_ids = build_model(config)
    counts = parameter_counts(model, pretrained_ids)
    log(f'parameters: {counts["total"]:,} total, {counts["trainable"]:,} trainable (heads {counts["heads"]:,}, new '
        f'backbone {counts["backbone_new"]:,}, unfrozen pretrained {counts["backbone_unfrozen"]:,})', log_file)
    if counts['total'] > MAX_TOTAL_PARAMETERS or counts['trainable'] > MAX_TRAINABLE_PARAMETERS:
        raise SystemExit(f'over the parameter budget (max {MAX_TOTAL_PARAMETERS:,} total, '
                         f'{MAX_TRAINABLE_PARAMETERS:,} trainable)')
    model = model.cuda()
    if run is not None:
        run.summary.update({'parameters': counts['total'], 'trainable_parameters': counts['trainable']})

    # AdamW groups: DoRA + Convpass at backbone_lr, the unfrozen patch embedding at unfrozen_lr, the image classifier
    # at cls_lr, every other new module (neck, stem, RPN, RoI head, input adapter) at lr. The last group's lr is logged.
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for name, p in model.named_parameters() if not name.startswith('backbone.')]
    cls_ids = {id(p) for p in model.cls_head.parameters()}
    groups = [(cfg['backbone_lr'], [p for p in backbone_params if id(p) not in pretrained_ids]),
              (cfg['unfrozen_lr'], [p for p in backbone_params if id(p) in pretrained_ids]),
              (cfg['cls_lr'], [p for p in head_params if id(p) in cls_ids]),
              (cfg['lr'], [p for p in head_params if id(p) not in cls_ids])]
    optimizer = torch.optim.AdamW([{'params': params, 'weight_decay': cfg['weight_decay'], 'lr': lr}
                                   for lr, params in groups], lr=cfg['lr'])
    warmup = max(1, round(cfg['warmup_epochs'] * len(train_loader)))
    total = cfg['epochs'] * len(train_loader)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, warmup, total))

    history, done, elapsed, start = [], 0, 0.0, time.time()  # done: last finished epoch; elapsed: earlier jobs
    if resume:
        state = load_checkpoint(last, model, optimizer, schedule)
        done, history, elapsed = state['epoch'], state['history'], state['elapsed']
        log(f'resumed from {last}: epoch {done} of {cfg["epochs"]} done, {elapsed / 3600:.2f} h so far', log_file)
    elif args.resume:
        log(f'--resume: no {last} yet, starting from scratch', log_file)

    for epoch in range(done + 1, cfg['epochs'] + 1):
        train_one_epoch(model, train_loader, optimizer, schedule, epoch, config['loss'], log_file, run)
        if val_loader is not None and (epoch % cfg['eval_every'] == 0 or epoch == cfg['epochs']):
            val = evaluate(model, val_loader)
            history.append({'epoch': epoch, **val})
            log(f'epoch {epoch} val: {headline(val)}', log_file)
            if run is not None:
                run.log({f'val/{k}': v for k, v in val.items() if not isinstance(v, (dict, list))},
                        step=epoch * len(train_loader))
        if cfg['checkpoint_every'] and (epoch % cfg['checkpoint_every'] == 0 or epoch == cfg['epochs']):
            save_checkpoint(last, model, optimizer, schedule, epoch, history, elapsed + time.time() - start,
                            extra={'wandb_id': run.id if run is not None else None})

    train_hours = (elapsed + time.time() - start) / 3600
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    log(f'training {train_hours:.2f} h, peak GPU memory {peak_gb:.1f} GB', log_file)
    summary = {'config': config, 'parameters': counts, 'train_hours': train_hours, 'peak_gpu_memory_gb': peak_gb,
               'val_history': history, 'val': history[-1] if history else None}
    (run_dir / 'metrics.json').write_text(json.dumps(summary, indent=2, default=str))
    torch.save(model.state_dict(), run_dir / 'model.pt')
    last.unlink(missing_ok=True)  # model.pt is the finished run; the checkpoint (~0.5 GB) is no longer needed
    if run is not None:
        run.summary.update({'train_hours': train_hours, 'peak_gpu_memory_gb': peak_gb})
        run.finish()


if __name__ == '__main__':
    main()
