"""Turn the CSVs from pull_wandb.py into report figures, written beside them in each ablation's folder.

Example: python plot_wandb.py --data ~/Desktop/cs701_report_data
         python plot_wandb.py --data ~/Desktop/cs701_report_data --format pdf

Needs matplotlib and pandas (not in requirements.txt: only this script uses them). Metrics are held-out, at default
post-processing (score 0.05, NMS 0.5), final epoch, one seed. A chart with a stopped run or a missing score is skipped;
add the scores to CODABENCH and score_narrow to draw the two Codabench charts.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

# Categorical order (validated with the dataviz skill's validate_palette.js); a parent run is neutral gray.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
PARENT = '#7c7b76'
MARKERS = ['o', 's', '^', 'D', 'v', 'P']
INK, MUTED, GRID, NOISE = '#0b0b0b', '#52514e', '#e4e3df', '#ecebe7'
UP, DOWN = '#2a78d6', '#eb6834'

METRICS = {'val/mAP': 'mAP', 'val/AP75': 'AP75', 'val/AP_small': 'AP_small', 'val/macro_f1': 'macro-F1'}
FOOT = 'Held-out split (939 images), default post-processing (score 0.05, NMS 0.5), one seed.'
SEED_NOISE = {'val/mAP': 0.06, 'val/AP75': 0.38, 'val/AP_small': 0.47, 'val/macro_f1': 0.07}  # |d1 - e3|, one pair

LABELS = {
    'dinov3_lora': 'LoRA, 24 ep', 'dinov3_moelora': 'MoE-LoRA, 24 ep', 'dinov3_dora': 'DoRA recipe, 48 ep',
    'dinov3_dora_giou': 'base (+ GIoU)', 'giou_a1_input': 'a1 input adapter',
    'giou_b1_full_blocks': 'b1 blocks 0-1 trained', 'giou_b2_rank_early': 'b2 rank 192 early',
    'giou_b3_rank_uniform': 'b3 rank 112', 'giou_b4_convpass': 'b4 Convpass 256',
    'giou_b5_hybrid': 'b5 hybrid', 'giou_c2_rank64_convpass192': 'c2 rank 64 + Convpass 192',
    'giou_b4_convpass_58ep': 'b4 58 ep', 'giou_b4_convpass_72ep': 'b4 72 ep',
    'giou_d3_head512_cp352': 'd3 head 512, Convpass 352', 'giou_d4_convhead_cp352': 'd4 conv head',
    'giou_d5_depthwise640': 'd5 depthwise 640', 'giou_d6_multiscale': 'd6 multiscale',
    'giou_d7_parallel': 'd7 parallel', 'giou_d9_stem128': 'd9 wide stem', 'giou_d1_96ep': 'd1 96 ep',
    'giou_e1_stem_96ep': 'e1 wide stem', 'giou_e2_120ep': 'e2 120 ep', 'giou_e3_d1_seed1': 'e3 seed 1',
    'giou_e4_stem_120ep': 'e4 120 ep', 'giou_e5_stem_150ep': 'e5 150 ep', 'giou_e6_stem_200ep': 'e6 200 ep',
    'giou_f1_adapter_lr3e-4': 'f1 adapters 3e-4', 'giou_f2_adapter_lr1e-3': 'f2 adapters 1e-3',
    'giou_f3_lr2e-4': 'f3 all 2e-4', 'giou_f5_split_lr': 'f5 split LR', 'giou_g1_split_lr_200ep': 'g1 split LR',
    'giou_h1_vit_adapter': 'h1 ViT-Adapter', 'giou_i1_cls_loss025_200ep': 'i1 weight 0.25',
    'giou_i2_cls_detach_200ep': 'i2 stop-gradient', 'giou_j0_base': 'j0 one level', 'giou_j1_roi_fused': 'j1 fused',
    'giou_j2_roi_canonical56': 'j2 canonical 56', 'giou_k1_copy_paste_small': 'k1 stronger paste',
    'giou_k4_giou_weight2': 'k4 GIoU 2.0', 'giou_k5_box_loss4': 'k5 box loss 4.0',
    'giou_k6_no_copy_paste': 'k6 no paste', 'giou_k7_alpha_giou': 'k7 alpha-GIoU',
    'giou_k11_roi_aligned': 'k11 aligned', 'giou_k19_bridge_stack': 'k19 BM3D', 'giou_k20_no_bm3d': 'k20 no BM3D',
    'giou_d10_full_train_e6': 'd10 all of train', 'giou_d11_full_train_g1_250ep': 'd11 weight 1.0',
    'giou_d12_full_train_g1': 'd12 all of train', 'giou_d13_full_train_i2_250ep': 'd13 stop-gradient',
    'giou_d14_full_train_i1_cls01_250ep': 'd14 weight 0.1', 'giou_d15_full_train_j0_250ep': 'd15 one level',
    'giou_d16_full_train_j1_250ep': 'd16 250 ep', 'giou_d17_full_train_j1_200ep': 'd17 200 ep',
}

# Codabench val delta-m recorded in the repo; None = not recorded yet, fill in to add the point.
CODABENCH = {'b4 58 ep': 19.16, 'b4 72 ep': 21.80, 'd1 96 ep': 26.65, 'e1 + wide stem': 28.33, 'e4 120 ep': 30.51,
             'e5 150 ep': 32.55, 'e6 200 ep': 35.39, 'g1 + split LR': 36.19, 'd10 e6 on all of train': 38.50,
             'd11 g1, 250 ep, all of train': 38.34, 'd14 classification 0.1': 43.58, 'd15 + augmentation fix': None,
             'd16 + fused RoI': None, 'd17 200 ep': None}


def style():
    plt.rcParams.update({
        'font.family': ['Helvetica', 'Arial', 'DejaVu Sans'], 'font.size': 9, 'axes.edgecolor': GRID,
        'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False, 'axes.grid': True,
        'axes.grid.axis': 'y', 'grid.color': GRID, 'grid.linewidth': 0.6, 'xtick.major.size': 0, 'ytick.major.size': 0,
        'axes.axisbelow': True, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'axes.titlesize': 9.5,
        'axes.titlecolor': MUTED, 'axes.titlelocation': 'left'})


def heading(fig, title, subtitle=None, foot=FOOT):
    fig.text(0.012, 0.975, title, fontsize=12.5, fontweight='bold', va='top')
    if subtitle:
        fig.text(0.012, 0.915, subtitle, fontsize=9.5, color=MUTED, va='top')
    if foot:
        fig.text(0.012, 0.012, foot, fontsize=7.5, color=MUTED, va='bottom')


def save(fig, folder, name, args):
    path = args.data / folder / f'{name}.{args.format}'
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(path.relative_to(args.data), flush=True)


def colours(runs, parent):
    """Run -> (colour, marker). The parent is gray; the others take the categorical slots in the given order."""
    slots = iter(zip(SERIES, MARKERS))
    return {run: (PARENT, 'X') if run == parent else next(slots) for run in runs}


def repel(ys, gap):
    """Spread label positions at least `gap` apart, keeping their order."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    placed = list(ys)
    for a, b in zip(order, order[1:]):
        placed[b] = max(placed[b], placed[a] + gap)
    shift = (sum(placed) - sum(ys)) / len(ys)
    return [y - shift for y in placed]


def complete(args, name, runs):
    """Whether every run finished: a chart built on a stopped or absent run is skipped, not drawn."""
    state = pd.read_csv(args.data / 'all_runs.csv').set_index('run').state
    incomplete = [run for run in runs if run not in state or state[run] != 'finished']
    if incomplete:
        print(f'skipped {name}: incomplete runs {incomplete}', flush=True)
    return not incomplete


def curves(args, folder, name, title, runs, parent=None, until=None, foot=FOOT):
    """Validation curves: mAP large, the other metrics small beside it. `until` compares and plots to that epoch."""
    if not complete(args, f'{folder}/{name}', runs):
        return
    val = pd.read_csv(args.data / folder / 'val.csv')
    val = val[val.run.isin(runs)]
    if until:
        val = val[val.epoch <= until]
    metrics = [m for m in METRICS if all(val[val.run == r][m].notna().any() for r in runs)]  # early runs lack AP_small
    style_of = colours(runs, parent)
    fig = plt.figure(figsize=(10.5, 4.1))
    # The narrow second column is empty: it holds the first panel's direct labels clear of the next axis.
    grid = fig.add_gridspec(1, len(metrics) + 1, width_ratios=[1.75, 0.2] + [1] * (len(metrics) - 1), left=0.05,
                            right=0.985, top=0.70, bottom=0.17, wspace=0.3)
    final = {r: val[val.run == r].sort_values('epoch').iloc[-1] for r in runs}
    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(grid[i if i == 0 else i + 1])
        shown = []
        for run in runs:
            frame = val[val.run == run].sort_values('epoch')
            colour, marker = style_of[run]
            ax.plot(frame.epoch, frame[metric] * 100, color=colour, lw=1.5, marker=marker, ms=4.5,
                    markevery=max(1, len(frame) // 6), markeredgecolor='white', markeredgewidth=0.5,
                    zorder=4 if run != parent else 3, solid_capstyle='round')
            late = frame[frame.epoch >= frame.epoch.max() * 0.3][metric] * 100
            shown += late.tolist()
        low, high = min(shown), max(shown)
        ax.set_ylim(low - (high - low) * 0.08, high + (high - low) * 0.1)
        ax.set_title(METRICS[metric] + (' (axis not from zero)' if i == 0 else ''), pad=5)
        ax.set_xlabel('epoch')
        if i == 0 and len(runs) <= 4:  # direct labels: name and final value, in text ink
            ends = [final[r][metric] * 100 for r in runs]
            span = ax.get_ylim()[1] - ax.get_ylim()[0]
            for run, y, end in zip(runs, repel(ends, span * 0.075), ends):
                ax.annotate(f'{LABELS[run].split()[0].rstrip(",")} {end:.1f}', xy=(1.012, y), xycoords=('axes fraction', 'data'),
                            fontsize=7.5, color=INK, va='center', annotation_clip=False)
    handles = []
    for run in runs:
        colour, marker = style_of[run]
        text = LABELS[run]
        last = val[val.run == run].epoch.max()
        if parent and run != parent:
            if run in partial_runs(args, folder) and not until:
                text += f' (stopped at epoch {int(last)})'
            else:
                text += f'  {(final[run]["val/mAP"] - final[parent]["val/mAP"]) * 100:+.2f} mAP'
        handles.append(Line2D([], [], color=colour, marker=marker, lw=1.5, ms=4, markeredgecolor='white', label=text))
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.006, 0.875), ncol=min(len(runs), 4), frameon=False,
               fontsize=8.5, handlelength=1.8, columnspacing=1.6)
    subtitle = None
    if parent and len(runs) == 2:
        other = next(r for r in runs if r != parent)
        subtitle = 'Change at the last shared epoch: ' + '   '.join(
            f'{METRICS[m]} {(final[other][m] - final[parent][m]) * 100:+.2f}' for m in metrics)
    heading(fig, title, subtitle, foot)
    save(fig, folder, name, args)


def partial_runs(args, folder):
    summary = pd.read_csv(args.data / folder / 'summary.csv')
    return set(summary[summary.state != 'finished'].run)


def deltas(args, folder, name, title, parent, runs, metrics=('val/mAP', 'val/macro_f1'), foot=None):
    """Final-epoch change against `parent`, one panel per metric, ranked by the first; gray band = seed noise."""
    if not complete(args, f'{folder}/{name}', [parent, *runs]):
        return
    final = pd.read_csv(args.data / 'all_runs.csv').set_index('run')
    order = sorted(runs, key=lambda r: final.loc[r, metrics[0]])
    fig, axes = plt.subplots(1, len(metrics), figsize=(10.5, 1.5 + 0.42 * len(runs)), sharey=True)
    axes = axes if len(metrics) > 1 else [axes]
    top = 1 - 0.85 / (1.5 + 0.42 * len(runs))
    fig.subplots_adjust(left=0.2, right=0.965, top=top, bottom=0.42 / (1.5 + 0.42 * len(runs)) + 0.05, wspace=0.14)
    for ax, metric in zip(axes, metrics):
        values = [(final.loc[r, metric] - final.loc[parent, metric]) * 100 for r in order]
        noise = SEED_NOISE[metric]
        ax.axvspan(-noise, noise, color=NOISE, lw=0, zorder=0)
        ax.barh(range(len(order)), values, height=0.5, color=[UP if v >= 0 else DOWN for v in values], zorder=3)
        ax.axvline(0, color=MUTED, lw=0.7, zorder=4)
        reach = max(abs(v) for v in values)
        ax.set_xlim(min(min(values), 0) - reach * 0.2, max(max(values), 0) + reach * 0.2)
        for y, v in enumerate(values):
            ax.text(v + (reach * 0.025 if v >= 0 else -reach * 0.025), y, f'{v:+.2f}', va='center', fontsize=8,
                    ha='left' if v >= 0 else 'right', color=INK, zorder=5)
        ax.set_yticks(range(len(order)), [LABELS[r] for r in order], color=INK)
        ax.set_title(f'{METRICS[metric]} change, points', pad=5)
        ax.grid(False)
        ax.tick_params(axis='x', labelsize=8)
    heading(fig, title, None, None)
    fig.text(0.012, 0.012, foot or f'Against {LABELS[parent]}, final epoch. Gray band: seed noise from one pair '
             f'(d1 vs e3). {FOOT}', fontsize=7.5, color=MUTED, va='bottom')
    save(fig, folder, name, args)


def losses(args, folder, name, title, runs, parent=None, keys=None, fraction=False, foot=None):
    """Smoothed raw training losses (no held-out data exists for full-train runs), one panel per loss term."""
    if not complete(args, f'{folder}/{name}', runs):
        return
    keys = keys or {'train/loss_box_reg': 'RoI box loss', 'train/loss_classifier': 'RoI class loss',
                    'train/loss_objectness': 'RPN objectness loss'}
    train = pd.read_csv(args.data / folder / 'train.csv')
    style_of = colours(runs, parent)
    fig, axes = plt.subplots(1, len(keys), figsize=(10.5, 3.7))
    axes = axes if len(keys) > 1 else [axes]
    fig.subplots_adjust(left=0.05, right=0.985, top=0.72, bottom=0.2, wspace=0.2)
    for ax, (key, label) in zip(axes, keys.items()):
        for run in runs:
            frame = train[train.run == run].sort_values('step')
            x = frame.epoch / frame.epoch.max() * 100 if fraction else frame.epoch
            ax.plot(x, frame[key].rolling(40, min_periods=5, center=True).median(), color=style_of[run][0], lw=1.5)
        ax.set_ylim(bottom=0)
        ax.set_title(label + ' (raw, unweighted)', pad=5)
        ax.set_xlabel('% of training' if fraction else 'epoch')
    handles = [Line2D([], [], color=style_of[r][0], lw=1.5, label=LABELS[r]) for r in runs]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.006, 0.9), ncol=len(runs), frameon=False,
               fontsize=8.5, handlelength=1.8)
    heading(fig, title, None, foot or 'Training set, median of 40 logged points. Lower training loss does not show '
            'better generalization. These runs have no held-out split.')
    save(fig, folder, name, args)


def schedule(args):
    """AB-15: what extra epochs bought, held-out and on Codabench, for the two series with a full ladder."""
    final = pd.read_csv(args.data / 'all_runs.csv').set_index('run')
    narrow = {48: 'giou_b4_convpass', 58: 'giou_b4_convpass_58ep', 72: 'giou_b4_convpass_72ep', 96: 'giou_d1_96ep',
              120: 'giou_e2_120ep'}
    wide = {96: 'giou_e1_stem_96ep', 120: 'giou_e4_stem_120ep', 150: 'giou_e5_stem_150ep', 200: 'giou_e6_stem_200ep'}
    score_narrow = {48: None, 58: 19.16, 72: 21.80, 96: 26.65, 120: 29.95}  # None: not recorded; fill in to draw
    score_wide = {96: 28.33, 120: 30.51, 150: 32.55, 200: 35.39}
    if None in score_narrow.values():
        print('skipped AB-15/schedule_ladder: Codabench score missing for 48 epochs', flush=True)
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.72, bottom=0.17, wspace=0.2)
    for ax, series, unit in ((axes[0], [(narrow, PARENT, 'X'), (wide, SERIES[0], 'o')], 'held-out mAP (axis not from zero)'),
                             (axes[1], [(score_narrow, PARENT, 'X'), (score_wide, SERIES[0], 'o')],
                              'Codabench val Δm (axis not from zero)')):
        for points, colour, marker in series:
            xs = list(points)
            ys = [final.loc[r, 'val/mAP'] * 100 if isinstance(r, str) else r for r in points.values()]
            ax.plot(xs, ys, color=colour, lw=1.5, marker=marker, ms=5, markeredgecolor='white', markeredgewidth=0.6)
            for x, y in zip(xs, ys):
                ax.annotate(f'{y:.1f}', (x, y), xytext=(0, 7 if colour != PARENT else -12), textcoords='offset points',
                            ha='center', fontsize=7.5, color=INK)
        low, high = ax.get_ylim()
        ax.set_ylim(low - (high - low) * 0.09, high + (high - low) * 0.06)
        ax.set_title(unit, pad=5)
        ax.set_xlabel('training epochs (each point is a separate run)')
    handles = [Line2D([], [], color=PARENT, marker='X', lw=1.5, ms=5, markeredgecolor='white', label='narrow stem'),
               Line2D([], [], color=SERIES[0], marker='o', lw=1.5, ms=5, markeredgecolor='white', label='wide stem')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.006, 0.9), ncol=2, frameon=False, fontsize=8.5)
    heading(fig, 'Every schedule extension paid, on both measures', None,
            'Default post-processing for held-out mAP. Codabench submissions: 0.001 / 0.6 where logged; values from the repo.')
    save(fig, 'AB-15', 'schedule_ladder', args)


def path_to_d16(args):
    """Overview: the held-out path, one row per step, in order."""
    final = pd.read_csv(args.data / 'all_runs.csv').set_index('run')
    steps = [('dinov3_lora', 'DINOv3, pseudo-RGB, rank 32 (24 ep)'), ('dinov3_dora', '+ DoRA, stem, aug, loss weights (48 ep)'),
             ('dinov3_dora_giou', '+ GIoU'), ('giou_a1_input', '+ input adapter'), ('giou_b4_convpass', '+ Convpass'),
             ('giou_d1_96ep', '+ schedule (96 ep)'), ('giou_e1_stem_96ep', '+ wide stem'),
             ('giou_f5_split_lr', '+ split learning rates'), ('giou_j0_base', '+ classification 0.1, augmentation fix'),
             ('giou_j1_roi_fused', '+ fused RoI pooling (j1: d16 recipe, 96 ep)')]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.6), sharey=True)
    fig.subplots_adjust(left=0.265, right=0.975, top=0.84, bottom=0.14, wspace=0.12)
    y = list(range(len(steps)))[::-1]
    for ax, metric in zip(axes, ('val/mAP', 'val/AP75', 'val/macro_f1')):
        values = [final.loc[run, metric] * 100 for run, _ in steps]
        ax.plot(values, y, color=SERIES[0], lw=1.5, marker='o', ms=5, markeredgecolor='white', markeredgewidth=0.6)
        for i, (v, row) in enumerate(zip(values, y)):  # a local minimum labels left, clear of its two line segments
            dip = 0 < i < len(values) - 1 and values[i - 1] > v < values[i + 1]
            ax.annotate(f'{v:.1f}', (v, row), xytext=(-7 if dip else 7, 0), textcoords='offset points', va='center',
                        ha='right' if dip else 'left', fontsize=7.5)
        ax.set_yticks(y, [label for _, label in steps], color=INK)
        ax.set_title(METRICS[metric] + ' (axis not from zero)', pad=5)
        ax.set_xlim(min(values) - (max(values) - min(values)) * 0.16, max(values) + (max(values) - min(values)) * 0.2)
        ax.grid(False)
        ax.grid(True, axis='x')
    heading(fig, 'Held-out path to the d16 recipe', None, FOOT + ' Each row adds to the one above. Code dates differ before a1.')
    save(fig, 'Overview', 'path_to_d16', args)


def codabench(args):
    """Overview: the recorded Codabench ladder; unrecorded runs are named in the footnote until filled in."""
    missing = [k.split()[0] for k, v in CODABENCH.items() if v is None]
    if missing:
        print(f'skipped Overview/codabench_ladder: no score for {", ".join(missing)}', flush=True)
        return
    known = CODABENCH
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    fig.subplots_adjust(left=0.2, right=0.965, top=0.86, bottom=0.14)
    y = list(range(len(known)))[::-1]
    ax.barh(y, list(known.values()), height=0.5, color=SERIES[0], zorder=3)
    for value, row in zip(known.values(), y):
        ax.text(value + 0.5, row, f'{value:.2f}', va='center', fontsize=8)
    ax.set_yticks(y, list(known), color=INK)
    ax.set_xlim(0, max(known.values()) * 1.1)
    ax.set_title('Codabench val Δm', pad=5)
    ax.grid(False)
    ax.grid(True, axis='x')
    heading(fig, 'Codabench scores recorded in the repo', None,
            'Post-processing differs between submissions; see section 7 of the seed.')
    save(fig, 'Overview', 'codabench_ladder', args)


def full_train_gain(args):
    """AB-17: the one clean Codabench pair."""
    fig, ax = plt.subplots(figsize=(6.5, 2.5))
    fig.subplots_adjust(left=0.3, right=0.93, top=0.68, bottom=0.2)
    ax.barh([1, 0], [35.39, 38.50], height=0.5, color=[PARENT, SERIES[0]], zorder=3)
    ax.set_yticks([1, 0], ['e6: 90% of train', 'd10: all of train'], color=INK)
    for row, value in ((1, 35.39), (0, 38.50)):
        ax.text(value + 0.6, row, f'{value:.2f}', va='center', fontsize=8.5)
    ax.set_xlim(0, 44)
    ax.grid(False)
    ax.set_title('Codabench val Δm', pad=5)
    heading(fig, 'Training on all of train: +3.11 Δm', None, 'Same recipe and logged post-processing. 11% more images and steps.')
    save(fig, 'AB-17', 'codabench_full_train', args)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--format', default='png', choices=['png', 'pdf', 'svg'])
    args = p.parse_args()
    args.data = args.data.expanduser()
    style()
    b = 'giou_'

    curves(args, 'AB-01 and AB-04', 'curves', 'Early DINOv3 runs: not a clean LoRA vs DoRA test',
           ['dinov3_dora', 'dinov3_moelora', 'dinov3_lora'],
           foot=FOOT + ' DoRA recipe also adds stem, augmentation, loss weights and 24 more epochs.')
    curves(args, 'AB-02', 'curves_partial', 'Without BM3D, stopped at epoch 21: partial evidence only',
           [b + 'k20_no_bm3d', b + 'k19_bridge_stack'], b + 'k19_bridge_stack', until=20,
           foot=FOOT + ' 20 of 96 epochs, k19 lineage rather than d16.')
    curves(args, 'AB-03', 'curves', 'Input adapter: no gain at its own step', [b + 'a1_input', 'dinov3_dora_giou'],
           'dinov3_dora_giou', foot=FOOT + ' The two runs differ by three days of code.')
    deltas(args, 'AB-05', 'rank_delta', 'Rank across every block beat rank in the early blocks', b + 'a1_input',
           [b + 'b2_rank_early', b + 'b3_rank_uniform', b + 'c2_rank64_convpass192'])
    curves(args, 'AB-05', 'curves', 'Higher-rank variants (context: rank 16 was never tested)',
           [b + 'b3_rank_uniform', b + 'b2_rank_early', b + 'c2_rank64_convpass192', b + 'a1_input'], b + 'a1_input')
    deltas(args, 'AB-06', 'equal_budget_delta', 'Equal-budget options: Convpass runs lead, and c2 scored highest',
           b + 'a1_input', [b + 'b1_full_blocks', b + 'b2_rank_early', b + 'b3_rank_uniform', b + 'b4_convpass',
                            b + 'b5_hybrid', b + 'c2_rank64_convpass192'])
    curves(args, 'AB-06', 'curves', 'Spatial adapters vs the best rank option',
           [b + 'b4_convpass', b + 'c2_rank64_convpass192', b + 'b3_rank_uniform', b + 'a1_input'], b + 'a1_input')
    deltas(args, 'AB-06', 'variants_delta', 'Convpass variants: none clearly beat the plain adapter', b + 'b4_convpass_72ep',
           [b + 'd3_head512_cp352', b + 'd4_convhead_cp352', b + 'd5_depthwise640', b + 'd6_multiscale', b + 'd7_parallel'])
    curves(args, 'AB-07', 'curves', 'P2 stem vs its only tested alternative', [b + 'h1_vit_adapter', b + 'f5_split_lr'],
           b + 'f5_split_lr', foot=FOOT + ' No run turns the stem off.')
    curves(args, 'AB-08', 'curves_72ep', 'Wider stem at 72 epochs', [b + 'd9_stem128', b + 'b4_convpass_72ep'],
           b + 'b4_convpass_72ep')
    curves(args, 'AB-08', 'curves_96ep', 'Wider stem at 96 epochs', [b + 'e1_stem_96ep', b + 'd1_96ep'], b + 'd1_96ep')
    curves(args, 'AB-09', 'curves', 'Fused RoI pooling lifts mAP and AP75',
           [b + 'j1_roi_fused', b + 'j2_roi_canonical56', b + 'k11_roi_aligned', b + 'j0_base'], b + 'j0_base')
    losses(args, 'AB-09', 'train_loss_full_length', 'Fused pooling at 250 epochs: training losses',
           [b + 'd16_full_train_j1_250ep', b + 'd15_full_train_j0_250ep'], b + 'd15_full_train_j0_250ep')
    curves(args, 'AB-10', 'curves', 'GIoU term: AP75 gains more than mAP', ['dinov3_dora_giou', 'dinov3_dora'], 'dinov3_dora',
           foot=FOOT + ' The two runs are from different code dates.')
    curves(args, 'AB-10', 'curves_on_j1', 'More GIoU did not help on the j1 recipe',
           [b + 'k4_giou_weight2', b + 'k7_alpha_giou', b + 'j1_roi_fused'], b + 'j1_roi_fused')
    curves(args, 'AB-11', 'curves', 'Raising the box loss weight further hurt', [b + 'k5_box_loss4', b + 'j1_roi_fused'],
           b + 'j1_roi_fused', foot=FOOT + ' No run tests 1.0, the reference value.')
    curves(args, 'AB-12', 'curves', 'Less classification gradient raised mAP',
           [b + 'i1_cls_loss025_200ep', b + 'i2_cls_detach_200ep', b + 'g1_split_lr_200ep'], b + 'g1_split_lr_200ep')
    losses(args, 'AB-12', 'train_loss_full_length', 'Classification weight at 250 epochs: training losses',
           [b + 'd14_full_train_i1_cls01_250ep', b + 'd13_full_train_i2_250ep', b + 'd11_full_train_g1_250ep'],
           b + 'd11_full_train_g1_250ep',
           keys={'train/classification': 'Image classification loss', 'train/loss_box_reg': 'RoI box loss',
                 'train/loss_classifier': 'RoI class loss'})
    curves(args, 'AB-13', 'curves', 'Copy-paste: small net gain, mainly at AP75',
           [b + 'k6_no_copy_paste', b + 'j1_roi_fused'], b + 'j1_roi_fused')
    curves(args, 'AB-13', 'curves_stronger_paste', 'Stronger pasting trailed from the start, and was stopped',
           [b + 'k1_copy_paste_small', b + 'j1_roi_fused'], b + 'j1_roi_fused', until=28)
    curves(args, 'AB-14', 'curves', 'Removing mosaic, in a bundle: the largest held-out jump',
           [b + 'j0_base', b + 'f5_split_lr'], b + 'f5_split_lr',
           foot=FOOT + ' j0 also changes classification weight and copy-paste.')
    losses(args, 'AB-14', 'train_loss_full_length', 'Augmentation fix at 250 epochs: training losses',
           [b + 'd15_full_train_j0_250ep', b + 'd14_full_train_i1_cls01_250ep'], b + 'd14_full_train_i1_cls01_250ep')
    schedule(args)
    curves(args, 'AB-15', 'curves_wide_stem', 'Longer cosine schedules keep rising to the end',
           [b + 'e6_stem_200ep', b + 'e5_stem_150ep', b + 'e4_stem_120ep', b + 'e1_stem_96ep'], b + 'e1_stem_96ep')
    losses(args, 'AB-15', 'train_loss_d16_vs_d17', 'd16 vs d17: 250 vs 200 epochs, training losses',
           [b + 'd16_full_train_j1_250ep', b + 'd17_full_train_j1_200ep'], b + 'd17_full_train_j1_200ep', fraction=True)
    deltas(args, 'AB-16', 'lr_delta', 'Split rates kept most of the mAP gain and most of the F1', b + 'e1_stem_96ep',
           [b + 'f1_adapter_lr3e-4', b + 'f2_adapter_lr1e-3', b + 'f3_lr2e-4', b + 'f5_split_lr'])
    curves(args, 'AB-16', 'curves_96ep', 'Learning rates at 96 epochs',
           [b + 'f5_split_lr', b + 'f3_lr2e-4', b + 'f1_adapter_lr3e-4', b + 'e1_stem_96ep'], b + 'e1_stem_96ep')
    curves(args, 'AB-16', 'curves_200ep', 'Split learning rates at 200 epochs', [b + 'g1_split_lr_200ep', b + 'e6_stem_200ep'],
           b + 'e6_stem_200ep')
    full_train_gain(args)
    losses(args, 'AB-17', 'train_loss', 'All of train vs 90%: training losses', [b + 'd10_full_train_e6', b + 'e6_stem_200ep'],
           b + 'e6_stem_200ep')
    curves(args, 'Seed noise', 'curves', 'Seed noise: d1 vs the same run at seed 1', [b + 'e3_d1_seed1', b + 'd1_96ep'],
           b + 'd1_96ep')
    path_to_d16(args)
    codabench(args)


if __name__ == '__main__':
    main()
