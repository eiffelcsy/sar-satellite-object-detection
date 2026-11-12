"""Download the W&B history of the ablation runs into one folder per ablation, ready for plotting.

Example: python pull_wandb.py --out ~/Desktop/cs701_report_data
         python pull_wandb.py --out ~/Desktop/cs701_report_data --only AB-09 AB-15

Writes, in <out>:
    all_runs.csv            every run in the project: state, final metrics, hours, GPU, the settings that matter,
                            and any setting that differs from the config file in the repo today
    <ablation>/val.csv      held-out metrics, one row per run and validation (every 4 epochs)
    <ablation>/train.csv    training losses, one row per run and logged step (every 50 steps)
    <ablation>/summary.csv  the all_runs.csv rows of that ablation's runs

Metrics are as logged during training: post-processing at the config defaults (score 0.05, NMS 0.5), not the tuned
values. Full-train runs (val_fraction 0) have no val rows. per_class_AP and the confusion matrix were never logged
to W&B: they are in runs/<name>/metrics.json on the cluster.
"""
import argparse
import csv
from pathlib import Path

import wandb

from sarbench.config import load_config

PROJECT = 'sar-satellite-object-detection'

ABLATIONS = {
    'AB-01 and AB-04': ['dinov3_lora', 'dinov3_moelora', 'dinov3_dora'],
    'AB-02': ['giou_k19_bridge_stack', 'giou_k20_no_bm3d'],
    'AB-03': ['dinov3_dora_giou', 'giou_a1_input'],
    'AB-05': ['giou_a1_input', 'giou_b2_rank_early', 'giou_b3_rank_uniform', 'giou_c2_rank64_convpass192',
              'giou_b4_convpass'],
    'AB-06': ['giou_a1_input', 'giou_b1_full_blocks', 'giou_b2_rank_early', 'giou_b3_rank_uniform', 'giou_b4_convpass',
              'giou_b5_hybrid', 'giou_c2_rank64_convpass192', 'giou_b4_convpass_72ep', 'giou_d3_head512_cp352',
              'giou_d4_convhead_cp352', 'giou_d5_depthwise640', 'giou_d6_multiscale', 'giou_d7_parallel'],
    'AB-07': ['giou_f5_split_lr', 'giou_h1_vit_adapter'],
    'AB-08': ['giou_b4_convpass_72ep', 'giou_d9_stem128', 'giou_d1_96ep', 'giou_e1_stem_96ep'],
    'AB-09': ['giou_j0_base', 'giou_j1_roi_fused', 'giou_j2_roi_canonical56', 'giou_k11_roi_aligned',
              'giou_d15_full_train_j0_250ep', 'giou_d16_full_train_j1_250ep'],
    'AB-10': ['dinov3_dora', 'dinov3_dora_giou', 'giou_j1_roi_fused', 'giou_k4_giou_weight2', 'giou_k7_alpha_giou'],
    'AB-11': ['giou_j1_roi_fused', 'giou_k5_box_loss4'],
    'AB-12': ['giou_g1_split_lr_200ep', 'giou_i1_cls_loss025_200ep', 'giou_i2_cls_detach_200ep',
              'giou_d11_full_train_g1_250ep', 'giou_d13_full_train_i2_250ep', 'giou_d14_full_train_i1_cls01_250ep'],
    'AB-13': ['giou_j1_roi_fused', 'giou_k6_no_copy_paste', 'giou_k1_copy_paste_small'],
    'AB-14': ['giou_f5_split_lr', 'giou_j0_base', 'giou_d14_full_train_i1_cls01_250ep',
              'giou_d15_full_train_j0_250ep'],
    'AB-15': ['giou_b4_convpass', 'giou_b4_convpass_58ep', 'giou_b4_convpass_72ep', 'giou_d1_96ep', 'giou_e2_120ep',
              'giou_e1_stem_96ep', 'giou_e4_stem_120ep', 'giou_e5_stem_150ep', 'giou_e6_stem_200ep',
              'giou_d12_full_train_g1', 'giou_d11_full_train_g1_250ep', 'giou_d17_full_train_j1_200ep',
              'giou_d16_full_train_j1_250ep'],
    'AB-16': ['giou_e1_stem_96ep', 'giou_f1_adapter_lr3e-4', 'giou_f2_adapter_lr1e-3', 'giou_f3_lr2e-4',
              'giou_f5_split_lr', 'giou_e6_stem_200ep', 'giou_g1_split_lr_200ep'],
    'AB-17': ['giou_e6_stem_200ep', 'giou_d10_full_train_e6', 'giou_g1_split_lr_200ep', 'giou_d12_full_train_g1'],
    'Seed noise': ['giou_d1_96ep', 'giou_e3_d1_seed1'],
}

VAL_KEYS = ['val/accuracy', 'val/macro_f1', 'val/balanced_accuracy', 'val/mAP', 'val/AP50', 'val/AP75', 'val/AP_small',
            'val/AP_medium', 'val/AP_large']
TRAIN_KEYS = ['train/loss', 'train/lr', 'train/classification', 'train/loss_classifier', 'train/loss_box_reg',
              'train/loss_objectness', 'train/loss_rpn_box_reg']
SETTINGS = ['epochs', 'val_fraction', 'batch_size', 'lr', 'backbone_lr', 'cls_lr', 'seed', 'mosaic', 'copy_paste',
            'copy_paste_same_class', 'copy_paste_margin', 'roi_pooling', 'giou_weight', 'loss_weights', 'detail_stem',
            'detail_stem_params', 'input_adapter', 'conv_adapters', 'peft_kwargs', 'peft_targets', 'pseudo_rgb',
            'score_thresh', 'nms_thresh']


def write_csv(path, rows):
    """Write dicts to `path`; the columns are the union of the rows' keys, in first-seen order."""
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows(rows)


def config_differences(run):
    """Settings where the run's logged config differs from its config file today, as 'key: logged -> file'."""
    path = Path(str(run.config.get('config')))
    if not path.exists():
        return 'config file not in repo'
    expected = load_config(path)
    skip = {'name', 'data', 'out', 'workers', 'wandb', 'preprocess_cache', 'backbone_lr', 'cls_lr'}
    plain = lambda value: {str(k): v for k, v in value.items()} if isinstance(value, dict) else value  # JSON keys
    differences = [f'{key}: {run.config[key]} -> {expected[key]}' for key in expected
                   if key in run.config and key not in skip and plain(run.config[key]) != plain(expected[key])]
    for key, default in (('backbone_lr', 'lr'), ('cls_lr', 'lr')):  # train.py resolves None to --lr before logging
        wanted = expected[key] if expected[key] is not None else expected[default]
        if key in run.config and run.config[key] != wanted:
            differences.append(f'{key}: {run.config[key]} -> {wanted}')
    return '; '.join(differences)


def summarize(run):
    """One all_runs.csv row: identity, final metrics, cost, settings and config drift."""
    summary = run.summary._json_dict
    try:
        gpu = run.metadata.get('gpu') if run.metadata else None
    except Exception:  # metadata is a separate file, absent on some runs
        gpu = None
    row = {'run': run.name, 'id': run.id, 'state': run.state, 'created_utc': run.created_at, 'last_step': summary.get('_step'),
           'train_hours': summary.get('train_hours'), 'peak_gpu_memory_gb': summary.get('peak_gpu_memory_gb'),
           'parameters': summary.get('parameters'), 'trainable_parameters': summary.get('trainable_parameters'),
           'gpu': gpu}
    row |= {key: summary.get(key) for key in VAL_KEYS}
    row |= {key: run.config.get(key) for key in SETTINGS}
    row['differs_from_config_file'] = config_differences(run)
    return row


def history(run):
    """(val rows, train rows) of `run`. A validation shares its _step with the epoch's last training row."""
    val, train = [], []
    for row in run.scan_history(page_size=10000):
        step, epoch = row.get('_step'), row.get('epoch')
        if row.get('val/mAP') is not None:
            val.append({'run': run.name, 'step': step, 'epoch': epoch, **{key: row.get(key) for key in VAL_KEYS}})
        if row.get('train/loss') is not None:
            train.append({'run': run.name, 'step': step, 'epoch': epoch, **{key: row.get(key) for key in TRAIN_KEYS}})
    return val, train


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--entity', help='W&B entity (default: your login\'s)')
    p.add_argument('--only', nargs='+', choices=list(ABLATIONS), help='these ablations only')
    args = p.parse_args()

    api = wandb.Api(timeout=120)
    runs = list(api.runs(f'{args.entity or api.default_entity}/{PROJECT}', order='+created_at', per_page=200))
    print(f'{len(runs)} runs in {PROJECT}', flush=True)

    best = {}  # run name -> the newest finished run of that name, else the newest
    for run in runs:
        if run.name not in best or run.state == 'finished' or best[run.name].state != 'finished':
            best[run.name] = run
    summaries = {run.id: summarize(run) for run in runs}
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / 'all_runs.csv', list(summaries.values()))

    histories = {}  # each run is downloaded once, however many ablations use it
    for ablation, names in ABLATIONS.items():
        if args.only and ablation not in args.only:
            continue
        missing = [name for name in names if name not in best]
        present = [best[name] for name in names if name in best]
        for run in present:
            if run.name not in histories:
                histories[run.name] = history(run)
                print(f'{run.name}: {len(histories[run.name][0])} val, {len(histories[run.name][1])} train rows '
                      f'({run.state})', flush=True)
        folder = args.out / ablation
        folder.mkdir(exist_ok=True)
        val = [row for run in present for row in histories[run.name][0]]
        train = [row for run in present for row in histories[run.name][1]]
        if val:
            write_csv(folder / 'val.csv', val)
        write_csv(folder / 'train.csv', train)
        write_csv(folder / 'summary.csv', [summaries[run.id] for run in present])
        unfinished = [run.name for run in present if run.state != 'finished']
        print(f'== {ablation}: {len(present)} runs' + (f', MISSING {missing}' if missing else '')
              + (f', NOT FINISHED {unfinished}' if unfinished else ''), flush=True)


if __name__ == '__main__':
    main()
