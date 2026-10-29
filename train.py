"""Train one model jointly for classification and detection, then score it on val and test.
Examples: python train.py --config configs/dinov3_dora_giou.yaml
          python train.py --config configs/ablations/b1_full_blocks.yaml"""
import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sarbench.channels import PseudoRGB
from sarbench.config import DEFAULTS, load_config
from sarbench.data import (SARImages, SARMultiTask, collate, collate_images, split_available,
                           to_original_xywh)
from sarbench.adapters import dora_up_weights
from sarbench.ema import ModelEMA
from sarbench.metrics import classification_metrics, detection_metrics
from sarbench.model import MAX_TOTAL_PARAMETERS, MAX_TRAINABLE_PARAMETERS, build_model, parameter_counts

from make_submission import write_submission


def parse_args(argv=None):
    """CLI flags override the values from --config; without --config the defaults are the reference protocol."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=Path)
    known, _ = pre.parse_known_args(argv)
    cfg = load_config(known.config) if known.config else dict(DEFAULTS)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=known.config,
                   help='YAML run config (see configs/); explicit CLI flags override its values')
    p.add_argument('--name', default=cfg['name'], help='run name (default: dinov3_dora)')
    p.add_argument('--data', type=Path, default=Path(cfg['data']) if cfg['data']
                   else Path(__file__).resolve().parents[1] / 'dataset' / 'SARFact-Course-20K')
    p.add_argument('--detail-stem', action='store_true', default=cfg['detail_stem'],
                   help='add a conv stem on the input to give the P2 level real stride-4 detail')
    p.add_argument('--fusion-layers', nargs='+', type=int, default=cfg['fusion_layers'],
                   help='ViT block indices whose tokens are fused before the neck (multi-layer FPN fusion)')
    p.add_argument('--input-adapter', action='store_true', default=cfg['input_adapter'],
                   help='ImageNet norm + learnable 1x1 conv (no-op init) + unfrozen DINOv3 patch embedding')
    p.add_argument('--wst-stem', action='store_true', default=cfg['wst_stem'],
                   help='add wavelet scattering (J=2, L=8 -> 81 ch at stride 4) detail to the neck P2 level')
    p.add_argument('--epochs', type=int, default=cfg['epochs'])
    p.add_argument('--stage-a-epochs', type=int, default=cfg['stage_a_epochs'],
                   help='freeze the backbone (incl. DoRA) for the first N epochs and train the heads only')
    p.add_argument('--stage-a-lr', type=float, default=cfg['stage_a_lr'], help='learning rate for stage A')
    p.add_argument('--batch-size', type=int, default=cfg['batch_size'])
    p.add_argument('--lr', type=float, default=cfg['lr'])
    p.add_argument('--backbone-lr', type=float, default=cfg['backbone_lr'],
                   help='for the new backbone parameters: DoRA and Convpass adapters (default: --lr)')
    p.add_argument('--lora-plus-ratio', type=float, default=cfg['lora_plus_ratio'],
                   help='LoRA+: the DoRA B (up) matrices at --backbone-lr x this ratio (1 = off)')
    p.add_argument('--cls-lr', type=float, default=cfg['cls_lr'],
                   help='for the image classifier head (default: --lr)')
    p.add_argument('--unfrozen-lr', type=float, default=cfg['unfrozen_lr'],
                   help='for pretrained backbone weights made trainable (patch embedding, peft.full_blocks)')
    p.add_argument('--weight-decay', type=float, default=cfg['weight_decay'])
    p.add_argument('--warmup-epochs', type=float, default=cfg['warmup_epochs'],
                   help='linear learning-rate warmup before the cosine decay')
    p.add_argument('--wd-exclude-1d', action='store_true', default=cfg['wd_exclude_1d'],
                   help='no weight decay on 1-D parameters (biases, norm weights, DoRA magnitudes, LayerScale)')
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
    p.add_argument('--copy-paste-margin', type=int, default=cfg['copy_paste_margin'],
                   help='pixels of context pasted around each copy-pasted object (shadow, smearing, wakes)')
    p.add_argument('--copy-paste-same-class', action='store_true', default=cfg['copy_paste_same_class'],
                   help='copy-paste donors from images of the same class only')
    p.add_argument('--scale-jitter', nargs=2, type=float, default=cfg['scale_jitter'],
                   help='train-time scale jitter range, e.g. --scale-jitter 0.9 1.1 (omit to disable)')
    p.add_argument('--rotation90', action='store_true', default=cfg['rotation90'],
                   help='train-time random 0/90/180/270-degree rotations')
    p.add_argument('--seed', type=int, default=cfg['seed'], help='training randomness (init, order, augmentation)')
    p.add_argument('--split-seed', type=int, default=cfg['split_seed'],
                   help='which images are held out for validation (keep it fixed to compare runs)')
    p.add_argument('--out', type=Path, default=Path(cfg['out']))
    p.add_argument('--load', type=Path, default=None,
                   help='load a saved model.pt state_dict, skip training, and only evaluate + write submission zips')
    p.add_argument('--predict', type=Path, default=None,
                   help='images.json of an unlabelled split: with --load, predict and write submission_<split>.zip '
                        '(no metrics; the images have no labels)')
    p.add_argument('--predict-image-dir', type=Path, default=None,
                   help='directory holding the --predict images (default: the images.json directory, then its images/)')
    p.add_argument('--score-thresh', type=float, default=cfg['score_thresh'],
                   help='inference: drop detections scoring below this (see sweep_postprocess.py)')
    p.add_argument('--nms-thresh', type=float, default=cfg['nms_thresh'], help='inference: per-class NMS IoU')
    p.add_argument('--detections-per-img', type=int, default=cfg['detections_per_img'],
                   help='inference: keep at most this many detections per image')
    p.add_argument('--rescore', choices=['off', 'soft', 'hard', 'merge'], default=cfg['rescore'],
                   help='inference: re-score boxes with the image classifier (see sarbench/heads.py)')
    p.add_argument('--rescore-power', type=float, default=cfg['rescore_power'],
                   help='inference, --rescore soft: p(c | box) * p(c | image) ** power')
    p.add_argument('--iou-alpha', type=float, default=cfg['iou_alpha'],
                   help='inference, model.iou_aware: rank detections by class score x predicted IoU ** alpha')
    p.add_argument('--predict-split', default='val',
                   help='name for the --predict output zip, submission_<predict-split>.zip')
    p.add_argument('--pseudo-rgb', action='store_true', default=cfg['pseudo_rgb'],
                   help='DINOv3 input: 1-channel SAR -> 3 channels (amplitude, despeckled base, edge map)')
    p.add_argument('--pseudo-rgb-mode', choices=['despeckle', 'amp_log_edge', 'amplitude'],
                   default=cfg['pseudo_rgb_mode'],
                   help='pseudo-RGB assembly: despeckle | amp_log_edge | amplitude (see sarbench/channels.py)')
    p.add_argument('--edge', choices=['sobel', 'highpass'], default=cfg['edge'],
                   help='with --pseudo-rgb: the channel-3 high-frequency map (default: sobel)')
    p.add_argument('--no-despeckle', action='store_true', default=cfg['no_despeckle'],
                   help='with --pseudo-rgb: skip SAR-BM3D despeckling')
    p.add_argument('--clip-percentile', type=float, default=cfg['clip_percentile'],
                   help='with --pseudo-rgb: %% of brightest pixels clipped before scaling to [0, 1]')
    p.add_argument('--preprocess-cache', type=Path,
                   default=Path(cfg['preprocess_cache']) if cfg['preprocess_cache'] else None,
                   help='with --pseudo-rgb: cache pre-processed images here (first pass fills it)')
    p.add_argument('--bm3d-sigma', type=float, default=cfg['bm3d_sigma'],
                   help='log-speckle std for SAR-BM3D (default: estimated per image)')
    p.add_argument('--bm3d-profile', default=cfg['bm3d_profile'],
                   help='bm3d package profile (np, refilter, vn, high, deb)')
    p.add_argument('--bm3d-threads', type=int, default=cfg['bm3d_threads'], help='bm3d package threads per worker')
    p.add_argument('--wandb', action='store_true', default=cfg['wandb'],
                   help='log loss/metric curves to Weights & Biases')
    p.add_argument('--wandb-project', default=cfg['wandb_project'])
    p.add_argument('--wandb-run-name', default=cfg['wandb_run_name'],
                   help='defaults to the run name')
    args = p.parse_args(argv)
    # Backbone, adapter and loss options come from the config only.
    for key in ('pretrained', 'backbone_kwargs', 'peft_kwargs', 'peft_targets', 'rank_overrides', 'full_blocks',
                'conv_adapters', 'giou_weight', 'roi_sampling_ratio', 'roi_output_size', 'wst_params',
                'loss_weights', 'head', 'head_params', 'detail_stem_params', 'rpn_fg_iou_thresh',
                'rpn_bg_iou_thresh', 'ema_decay', 'vit_adapter', 'box_similarity', 'nwd_constant', 'iou_aware',
                'heatmap_aux', 'cls_detach', 'cls_input', 'roi_pooling', 'roi_canonical_scale',
                'roi_batch_size_per_image', 'roi_positive_fraction', 'copy_paste_small'):
        setattr(args, key, cfg[key])
    if args.predict is not None and args.load is None:
        p.error('--predict needs --load with the trained weights')
    if args.backbone_lr is None:
        args.backbone_lr = args.lr
    if args.cls_lr is None:
        args.cls_lr = args.lr
    return args


def log(message, file):
    """Print to the console and to the run's log.txt."""
    print(message, flush=True)
    print(message, file=file, flush=True)


def run_name(args):
    """The config's run name, or dinov3_dora when it is absent."""
    return args.name or 'dinov3_dora'


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
    """The pseudo-RGB channel assembly, or None (the gray image is repeated to RGB) when --pseudo-rgb is off."""
    if not args.pseudo_rgb:
        return None
    return PseudoRGB(despeckle=not args.no_despeckle, edge=args.edge, mode=args.pseudo_rgb_mode,
                     clip_percentile=args.clip_percentile, sigma=args.bm3d_sigma,
                     profile=args.bm3d_profile, threads=args.bm3d_threads)


def make_loader(args, split):
    """Batches of `split`; only the training set is shuffled and augmented."""
    train = split == 'train'
    dataset = SARMultiTask(args.data, split, train=train, limit=args.limit,
                           val_fraction=args.val_fraction, seed=args.split_seed,
                           preprocess=build_preprocess(args), cache_dir=args.preprocess_cache,
                           mosaic=args.mosaic, copy_paste=args.copy_paste,
                           copy_paste_margin=args.copy_paste_margin, copy_paste_same_class=args.copy_paste_same_class,
                           copy_paste_small=args.copy_paste_small,
                           scale_jitter=args.scale_jitter, rotation90=args.rotation90)
    return DataLoader(dataset, args.batch_size, shuffle=train, num_workers=args.workers, collate_fn=collate,
                      persistent_workers=train and args.workers > 0)  # training workers live across epochs


def make_prediction_loader(args):
    """Batches of the unlabelled images.json in --predict, using the same input pipeline as training."""
    dataset = SARImages(args.predict, args.predict_image_dir, preprocess=build_preprocess(args),
                        cache_dir=args.preprocess_cache)
    return DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.workers,
                      collate_fn=collate_images)


def lr_factor(step, warmup, total):
    """Learning-rate multiplier: linear warmup over `warmup` steps, then cosine decay to 0 at step `total`."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))  # --epochs 1: no decay phase


def decay_groups(params, weight_decay, exclude_1d, **options):
    """AdamW parameter groups for `params`: one group with `weight_decay`, or, with `exclude_1d`, the 1-D parameters
    (biases, norm weights, DoRA magnitudes, LayerScale) split into a second group without decay. Extra `options`
    (e.g. lr) go to every group."""
    params = list(params)
    if not exclude_1d:
        return [{'params': params, 'weight_decay': weight_decay, **options}]
    return [{'params': [p for p in params if p.ndim > 1], 'weight_decay': weight_decay, **options},
            {'params': [p for p in params if p.ndim <= 1], 'weight_decay': 0.0, **options}]


def weighted_total(losses, loss_weights):
    """Sum the loss terms, scaling each by its config weight (default 1.0)."""
    weights = loss_weights or {}
    return sum(weights.get(name, 1.0) * value for name, value in losses.items())


def train_one_epoch(model, loader, optimizer, schedule, epoch, log_file, run=None, loss_weights=None, ema=None):
    """One pass over the training set; the loss is classification plus the detector losses, weighted per term.
    With `ema`, the moving average is updated after every optimizer step."""
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
        if ema is not None:
            ema.update()
        seen += len(images)
        if step % 50 == 0 or step == len(loader):
            parts = ' '.join(f'{name} {value.item():.3f}' for name, value in losses.items())
            log(f'epoch {epoch} iter {step}/{len(loader)} loss {loss.item():.3f} (weighted total; raw: {parts}) '
                f'lr {optimizer.param_groups[-1]["lr"]:.2e} {seen / (time.time() - start):.1f} images/s', log_file)
            if run is not None:
                run.log({f'train/{name}': value.item() for name, value in losses.items()}
                        | {'train/loss': loss.item(), 'train/lr': optimizer.param_groups[-1]['lr'],
                           'epoch': epoch},
                        step=(epoch - 1) * len(loader) + step)
            start, seen = time.time(), 0


def fit(model, train_loader, val_loader, epochs, optimizer, schedule, args, log_file, run, history, start_epoch=1,
        ema=None):
    """Train `epochs` epochs, validating every --eval-every and after the last one; appends to `history`.

    Validation uses the EMA weights when `ema` is given, and is skipped without a `val_loader` (val_fraction 0).
    Returns the last validation's (detections, submission records), which the caller saves.
    """
    val_detections = val_sub = None
    for offset in range(epochs):
        epoch = start_epoch + offset
        train_one_epoch(model, train_loader, optimizer, schedule, epoch, log_file, run, args.loss_weights, ema)
        if val_loader is not None and (epoch % args.eval_every == 0 or offset == epochs - 1):
            with ema.averaged() if ema is not None else contextlib.nullcontext():
                val, val_detections, val_sub = evaluate(model, val_loader, val_loader.dataset.instances_json)
            history.append({'epoch': epoch, **val})
            log(f'epoch {epoch} val: {headline(val)}', log_file)
            if run is not None:
                run.log({f'val/{k}': v for k, v in val.items() if not isinstance(v, (dict, list))},
                        step=epoch * len(train_loader))
    return val_detections, val_sub


@torch.no_grad()
def evaluate(model, loader, gt_json):
    """Classification and COCO detection metrics on one split, the detections in COCO result format, and the
    per-image (label, raw input-space boxes, 0-indexed classes, scores) records that `write_submission` needs."""
    model.eval()
    labels, predicted, detections, image_ids = [], [], [], []
    sub_images, sub_preds = [], {}
    for images, y, targets in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, outputs, _ = model(images.cuda())
        batch_labels = logits.argmax(dim=1).tolist()
        labels += y.tolist()
        predicted += batch_labels
        for target, out, label in zip(targets, outputs, batch_labels):
            image_id = target['image_id']
            height, width = target['orig_size']
            image_ids.append(image_id)
            boxes = to_original_xywh(out['boxes'], target['orig_size']).tolist()
            detections += [{'image_id': image_id, 'category_id': c, 'bbox': b, 'score': s}
                           for b, c, s in zip(boxes, out['labels'].tolist(), out['scores'].tolist())]
            # write_submission wants the raw model-input boxes and 0-indexed classes (it adds 1 itself)
            key = str(image_id)
            sub_images.append({'id': image_id, 'file_name': key, 'width': width, 'height': height})
            sub_preds[key] = (label, out['boxes'].tolist(), (out['labels'] - 1).tolist(),
                              out['scores'].tolist())
    metrics = {**classification_metrics(labels, predicted), **detection_metrics(gt_json, detections, image_ids)}
    return metrics, detections, (sub_images, sub_preds)


@torch.no_grad()
def predict(model, loader):
    """Run the model over an images-only loader and return the (images, preds) pair for `write_submission`.

    No labels or ground truth are involved: the image id and original size come from images.json, and the
    predicted class, boxes, classes and scores from the model, so this works on the unlabelled val/test splits.
    """
    model.eval()
    sub_images, sub_preds = [], {}
    for images, image_ids, orig_sizes, file_names in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, outputs, _ = model(images.cuda())
        batch_labels = logits.argmax(dim=1).tolist()
        for image_id, orig_size, file_name, out, label in zip(image_ids, orig_sizes, file_names, outputs,
                                                              batch_labels):
            height, width = int(orig_size[0]), int(orig_size[1])
            sub_images.append({'id': image_id, 'file_name': file_name, 'width': width, 'height': height})
            sub_preds[file_name] = (label, out['boxes'].tolist(), (out['labels'] - 1).tolist(),
                                    out['scores'].tolist())
    return sub_images, sub_preds


def headline(metrics):
    return ' '.join(f'{k} {metrics[k]:.4f}' for k in ('accuracy', 'macro_f1', 'balanced_accuracy', 'mAP', 'AP50',
                                                      'AP75', 'AP_small', 'AP_medium', 'AP_large'))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    # TF32 for fp32 matmuls/convs (the RPN and RoI heads): importing terratorch used to switch this on implicitly,
    # so it is kept explicitly to leave speed and numerics unchanged.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run_dir = args.out / run_name(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(run_dir / 'log.txt', 'w')
    log(json.dumps(vars(args), default=str), log_file)
    run = start_wandb(args, run_dir)

    # Data
    if args.pseudo_rgb and args.preprocess_cache is None:
        log('--pseudo-rgb without --preprocess-cache: SAR-BM3D runs every epoch (slow); '
            'pass --preprocess-cache DIR to compute it once', log_file)
    if args.predict is not None:  # submission-only: no labelled split is needed
        train_loader = val_loader = test_loader = None
    else:
        train_loader = make_loader(args, 'train')
        val_loader = make_loader(args, 'val') if args.val_fraction > 0 else None  # 0: train on all of train
        if val_loader is None:
            log('val_fraction 0: training on all of train, no held-out validation', log_file)
        test_loader = make_loader(args, 'test') if split_available(args.data, 'test') else None
        if test_loader is None:
            log('no labelled test split found; skipping test evaluation', log_file)

    # Model: adapters are created on the CPU, so they are added before .cuda()
    model, pretrained_ids = build_model(args)
    counts = parameter_counts(model, pretrained_ids)
    parameters, trainable = counts['total'], counts['trainable']
    log(f'parameters: {parameters:,} total, {trainable:,} trainable (heads {counts["heads"]:,}, new backbone '
        f'{counts["backbone_new"]:,}, unfrozen pretrained {counts["backbone_unfrozen"]:,})', log_file)
    if parameters > MAX_TOTAL_PARAMETERS or trainable > MAX_TRAINABLE_PARAMETERS:
        raise SystemExit(f'over the parameter budget: {parameters:,} total (max {MAX_TOTAL_PARAMETERS:,}), '
                         f'{trainable:,} trainable (max {MAX_TRAINABLE_PARAMETERS:,})')
    model = model.cuda()
    if run is not None:
        run.summary.update({'parameters': parameters, 'trainable_parameters': trainable})

    # Submission-only: predict on an unlabelled images.json and write the zip (no labels, no metrics)
    if args.predict is not None:
        model.load_state_dict(torch.load(args.load, map_location='cuda'))
        log(f'loading weights from {args.load}; predicting on {args.predict}', log_file)
        sub_images, sub_preds = predict(model, make_prediction_loader(args))
        out = run_dir / f'submission_{args.predict_split}.zip'
        write_submission(sub_images, sub_preds, out=out)
        log(f'wrote {out} ({len(sub_images)} images)', log_file)
        if run is not None:
            run.finish()
        return

    # Inference-only: load saved weights and write submission zips for val (and test when labelled)
    if args.load is not None:
        if val_loader is None:
            raise SystemExit('--load evaluates on the held-out split; use val_fraction > 0, or --predict')
        model.load_state_dict(torch.load(args.load, map_location='cuda'))
        log(f'loaded weights from {args.load}', log_file)
        val, val_detections, val_sub = evaluate(model, val_loader, val_loader.dataset.instances_json)
        log(f'val: {headline(val)}', log_file)
        (run_dir / 'predictions_val.json').write_text(json.dumps(val_detections))
        write_submission(*val_sub, out=run_dir / 'submission_val.zip')
        if test_loader is not None:
            test, test_detections, test_sub = evaluate(model, test_loader, test_loader.dataset.instances_json)
            log(f'test: {headline(test)}', log_file)
            (run_dir / 'predictions_test.json').write_text(json.dumps(test_detections))
            write_submission(*test_sub, out=run_dir / 'submission_test.zip')
        log(f'wrote submission zip(s) to {run_dir}', log_file)
        if run is not None:
            run.finish()
        return

    # Four parameter groups: new backbone parameters (DoRA, Convpass) at --backbone-lr, pretrained weights made
    # trainable (patch embedding, full_blocks) gently at --unfrozen-lr, the image classifier at --cls-lr, and every
    # other new module outside the backbone (neck, RPN, RoI head, stems, input adapter) at --lr. With
    # --lora-plus-ratio, the DoRA B matrices get a fifth group at --backbone-lr x that ratio.
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    new_backbone = [p for p in backbone_params if id(p) not in pretrained_ids]
    lora_up_ids = {id(p) for p in dora_up_weights(model.backbone)} if args.lora_plus_ratio != 1 else set()
    lora_up = [p for p in new_backbone if id(p) in lora_up_ids]  # LoRA+: B at backbone_lr x lora_plus_ratio
    new_backbone = [p for p in new_backbone if id(p) not in lora_up_ids]
    unfrozen = [p for p in backbone_params if id(p) in pretrained_ids]
    head_params = [p for name, p in model.named_parameters() if not name.startswith('backbone.')]
    cls_ids = {id(p) for p in model.cls_head.parameters()}
    cls_params = [p for p in head_params if id(p) in cls_ids]
    det_params = [p for p in head_params if id(p) not in cls_ids]

    # Stage A: freeze the backbone (including DoRA) and train only the neck, RPN, RoI and class heads.
    history, start = [], time.time()
    if args.stage_a_epochs:
        log(f'stage A: {args.stage_a_epochs} epochs, backbone frozen, lr {args.stage_a_lr}', log_file)
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        optimizer_a = torch.optim.AdamW(decay_groups(head_params, args.weight_decay, args.wd_exclude_1d),
                                        lr=args.stage_a_lr)
        schedule_a = torch.optim.lr_scheduler.LambdaLR(
            optimizer_a, lambda step: lr_factor(step, len(train_loader), args.stage_a_epochs * len(train_loader)))
        fit(model, train_loader, val_loader, args.stage_a_epochs, optimizer_a, schedule_a, args, log_file, run,
            history)  # stage A's validations go into the history; the final ones come from stage B
        for p in backbone_params:  # switch DoRA/backbone back on for stage B
            p.requires_grad_(True)

    # Stage B (and the only stage when stage_a_epochs is 0): trainable backbone at --backbone-lr, heads at --lr.
    groups = (decay_groups(new_backbone, args.weight_decay, args.wd_exclude_1d, lr=args.backbone_lr)
              + decay_groups(lora_up, args.weight_decay, args.wd_exclude_1d,
                             lr=args.backbone_lr * args.lora_plus_ratio)
              + decay_groups(unfrozen, args.weight_decay, args.wd_exclude_1d, lr=args.unfrozen_lr)
              + decay_groups(cls_params, args.weight_decay, args.wd_exclude_1d, lr=args.cls_lr)
              + decay_groups(det_params, args.weight_decay, args.wd_exclude_1d, lr=args.lr))
    # the heads stay last: train_one_epoch logs param_groups[-1]'s lr
    optimizer = torch.optim.AdamW([g for g in groups if g['params']], lr=args.lr)
    warmup = max(1, round(args.warmup_epochs * len(train_loader)))
    total = args.epochs * len(train_loader)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, warmup, total))
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None  # over stage B's trainable weights
    val_detections, val_sub = fit(model, train_loader, val_loader, args.epochs, optimizer, schedule, args,
                                  log_file, run, history, start_epoch=args.stage_a_epochs + 1, ema=ema)
    train_hours = (time.time() - start) / 3600
    if ema is not None:  # the averaged weights are the final model: test and model.pt use them
        ema.copy_to_model()
        log(f'EMA (decay {args.ema_decay}) weights copied into the model', log_file)

    # Evaluate on test, once (the course val/test labels are withheld, so test may be absent)
    if test_loader is not None:
        test, test_detections, test_sub = evaluate(model, test_loader, test_loader.dataset.instances_json)
        log(f'test: {headline(test)}', log_file)
    else:
        test, test_detections, test_sub = None, None, None
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    log(f'training {train_hours:.2f} h, peak GPU memory {peak_gb:.1f} GB', log_file)
    if run is not None:
        if test is not None:
            run.log({f'test/{k}': v for k, v in test.items() if not isinstance(v, (dict, list))})
        run.summary.update({'train_hours': train_hours, 'peak_gpu_memory_gb': peak_gb})

    # Save
    summary = {'args': vars(args), 'parameters': parameters, 'trainable_parameters': trainable,
               'parameter_breakdown': counts,
               'train_hours': train_hours, 'peak_gpu_memory_gb': peak_gb,
               'val_history': history, 'val': history[-1] if history else None, 'test': test}
    (run_dir / 'metrics.json').write_text(json.dumps(summary, indent=2, default=str))
    if val_sub is not None:
        (run_dir / 'predictions_val.json').write_text(json.dumps(val_detections))
        write_submission(*val_sub, out=run_dir / 'submission_val.zip')
    if test_detections is not None:
        (run_dir / 'predictions_test.json').write_text(json.dumps(test_detections))
    if test_sub is not None:
        write_submission(*test_sub, out=run_dir / 'submission_test.zip')
    torch.save(model.state_dict(), run_dir / 'model.pt')
    if run is not None:
        run.finish()


if __name__ == '__main__':
    main()
