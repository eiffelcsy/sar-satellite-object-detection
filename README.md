# CS701 SAR Multi-Task Benchmark

Reference code for the CS701 team assignment. One ViT-B/16 backbone, two tasks on SAR images: **classify** the
image (9 classes) and **detect** its objects.

![One SAR image, one shared backbone, two heads: a class and the boxes](docs/task_pipeline.png)

## Contents

1. [SAR Task Introduction](#1-sar-task-introduction)
   - [1.1 SAR Basics](#11-sar-basics)
   - [1.2 Task](#12-task)
   - [1.3 Metric](#13-metric)
   - [1.4 Small Objects](#14-small-objects)
2. [Quick Start](#2-quick-start)
   - [2.1 Install](#21-install)
   - [2.2 Train](#22-train)
   - [2.3 Use Course Data](#23-use-course-data)
   - [2.4 Make Submission](#24-make-submission)
3. [Baseline Results (for reference)](#3-baseline-results-for-reference)
4. [Code Structure](#4-code-structure)
5. [Links](#5-links)

## 1. SAR Task Introduction

### 1.1 SAR Basics

![A radar sends microwave pulses sideways, records the echoes, and combines them along its flight path into one long virtual antenna](docs/sar_explainer.gif)

- **Active radar**: satellite or aircraft sends microwave pulses, records echoes. Works day, night, through clouds.
  Looks sideways.
- **Synthetic aperture**: echoes along the flight path combined into one long virtual antenna, so fine resolution.
- **Pixel = echo strength**: 8-bit grey, no colour. Metal and corners bright, calm water dark, grainy speckle.

<details>
<summary>What SAR images look like</summary>

![Speckle, bright metal and corners, dark smooth surfaces, bright and dark slopes](docs/sar_effects.png)

</details>

### 1.2 Task

- Input: one SAR image. Output: its class + a box, class and score for each object.
- Each image: one class label, 1 or more boxes, all of that class.
- 9 classes: aircraft, airport, bridge, car, harbor, oil tank, playground, ship, wind turbine.
- Data, formats, release dates: [dataset card](https://huggingface.co/datasets/doem1997/cs701-sar-course-data).
  Rules: [Codabench](https://www.codabench.org/competitions/18271).
- Limits: one model; at most 130 M parameters in total (frozen ones count; this code: 107-125 M) and at most 40 M
  trainable (heads included: the base 25.6 M, the capacity ablations up to 39.5 M); each image in once at
  ≤ 512 × 512 px; no ensembles, no test-time augmentation. Your report states parameters (total, trainable) and
  compute; full list on Codabench (Terms).

### 1.3 Metric

- Classification: macro-F1. Detection: COCO mAP.
- Ranking: **Δm**, mean relative change of macro-F1 and mAP vs a reference model (the course's ViT-B/16 + LoRA,
  section 3; mean of 3 seeds):
  `Δm = 100% × ½ [(F1 − F1_ref) / F1_ref + (mAP − mAP_ref) / mAP_ref]`
- Δm = 0: as good as the reference. Δm > 0: better. ViT full fine-tuning: about +13 (over the trainable budget: an
  upper bound).

### 1.4 Small Objects

Half of all boxes are smaller than one 16 × 16 ViT patch (at 512 × 512 input).

![A close-up with the 16 px patch grid: wind turbines of 6-10 px and the 15.9 px median box are smaller than one patch](docs/patch_grid.png)

## 2. Quick Start

### 2.1 Install

```bash
conda create -n cs701bench python=3.12 -y && conda activate cs701bench
pip install -r requirements.txt   # CUDA 12.8 builds of torch 2.9.0 / torchvision 0.24.0
```

### 2.2 Train

```bash
python train.py --config configs/dinov3_dora_giou.yaml   # the current best recipe (the base)
bash run_all.sh                                          # the base and every ablation in configs/ablations/
```

The run is configured by a YAML file under `configs/` (schema in `sarbench/config.py`); the command line stays
available for one-off overrides, e.g. `python train.py --config configs/dinov3_dora_giou.yaml --epochs 2 --limit 64`.
An ablation config names its parent with `base:` and lists only what it changes.

**Base: `configs/dinov3_dora_giou.yaml`.** DINOv3 ViT-B/16 + pseudo-RGB input (amplitude, SAR-BM3D despeckled,
Sobel edge) + **DoRA** r = 32 on attention and MLP + a real-detail **P2 conv stem** + a **GIoU** box term +
mosaic/copy-paste, 48 epochs. 111.27 M parameters in total, 25.61 M trainable.

**Ablations (`configs/ablations/`)**, each one change, with its exact parameter count:

| config | change | parent | total (M) | trainable (M) |
|---|---|---|---:|---:|
| `a1_input` | ImageNet norm + learnable 1x1 conv + unfrozen patch embedding | base | 111.27 | 26.20 |
| `a2_aug` | exact 90° rotations, scale jitter 0.9-1.1, 3 px copy-paste context | base | 111.27 | 25.61 |
| `a3_wst` | wavelet scattering (MSFA's filter input) added to P2 | base | 111.89 | 26.23 |
| `a4_warmstart` | 10 head-only epochs before the 48 (MSFA's model gap) | base | 111.27 | 25.61 |
| `a5_fusion` | learnable fusion of ViT blocks 5, 8, 11 | base | 111.27 | 25.61 |
| `a6_warmstart_cascade` | Cascade R-CNN (3 x 256-wide heads, IoU 0.4/0.5/0.6) | a4 | 107.20 | 21.54 |
| `b1_full_blocks` | blocks 0-1 trained entirely | a1 | 110.38 | 39.48 |
| `b2_rank_early` | DoRA rank 192 in blocks 0-5 | a1 | 124.55 | 39.48 |
| `b3_rank_uniform` | DoRA rank 112 in every block | a1 | 124.55 | 39.48 |
| `b4_convpass` | 256-wide Convpass adapter after every block | a1 | 123.09 | 38.02 |
| `b5_hybrid` | block 0 entirely + rank 96 in blocks 1-5 + Convpass in blocks 0-5 | a1 | 117.32 | 39.34 |
| `b4_convpass_58ep` | b4 trained 58 epochs (control for c1a/c1b) | b4 | 123.09 | 38.02 |
| `c1a_b4_warmstart` | b4 + 10 head-only epochs (a4) | b4 | 123.09 | 38.02 |
| `c1b_b4_warmstart_cascade` | c1a + Cascade R-CNN (a6) | c1a | 119.01 | 33.94 |
| `c2_rank64_convpass192` | DoRA rank 64 + 192-wide Convpass in every block | a1 | 124.12 | 39.05 |
| `b4_convpass_72ep` | b4 trained 72 epochs | b4 | 123.09 | 38.02 |
| `b4_wst_72ep` | b4_convpass_72ep + WST stem (a3) | b4_72ep | 123.70 | 38.63 |
| `d1_96ep` | b4_72ep trained 96 epochs | b4_72ep | 123.09 | 38.02 |
| `d2_ema` | EMA of the trainable weights (decay 0.9998), saved as the model | b4_72ep | 123.09 | 38.02 |
| `d3_head512_cp352` | box head 1024 -> 512 wide, the freed budget to 352-wide Convpass | b4_72ep | 123.93 | 38.86 |
| `d4_convhead_cp352` | 4-conv + 1-fc box head (5.6 M instead of 13.9 M) + 352-wide Convpass | b4_72ep | 122.86 | 37.79 |
| `d5_depthwise640` | depthwise 3x3 Convpass, bottleneck 256 -> 640 | b4_72ep | 123.16 | 38.09 |
| `d6_multiscale` | Convpass 3x3 split into dilation 1 + 2 halves | b4_72ep | 123.09 | 38.02 |
| `d7_parallel` | Convpass beside the block (reads its LayerNorm'd input) | b4_72ep | 123.10 | 38.04 |
| `d8_rpn06` | RPN positive anchor IoU 0.7 -> 0.6 | b4_72ep | 123.09 | 38.02 |
| `d9_stem128` | P2 conv stem 64 -> 128 channels + one extra stride-4 3x3 conv | b4_72ep | 123.93 | 38.86 |
| `d10_full_train` | final model: the e6 recipe on all of train (`val_fraction: 0`) | e6 | 123.93 | 38.86 |
| `e1_stem_96ep` | d1 + d9's stride-4 stem | d1 | 123.93 | 38.86 |
| `e2_120ep` | d1 trained 120 epochs | d1 | 123.09 | 38.02 |
| `e3_d1_seed1` | d1 with training seed 1 (same held-out images): the seed-noise check | d1 | 123.09 | 38.02 |
| `e4_stem_120ep` | e1 (d1 + stem) trained 120 epochs | e1 | 123.93 | 38.86 |
| `e5_stem_150ep` | e4 trained 150 epochs | e4 | 123.93 | 38.86 |
| `e6_stem_200ep` | e5 trained 200 epochs | e5 | 123.93 | 38.86 |
| `f1_adapter_lr3e-4` | adapters (DoRA, Convpass) at 3e-4 instead of 1e-4 | e1 | 123.93 | 38.86 |
| `f2_adapter_lr1e-3` | adapters at 1e-3 + a 2-epoch warmup | e1 | 123.93 | 38.86 |
| `f3_lr2e-4` | heads and adapters at 2e-4 | e1 | 123.93 | 38.86 |
| `f4_wd_exclude_1d` | no weight decay on biases, norm weights and DoRA magnitudes | e1 | 123.93 | 38.86 |
| `f5_split_lr` | adapters 3e-4, detection heads 2e-4, image classifier 1e-4 | e1 | 123.93 | 38.86 |
| `g1_split_lr_200ep` | f5's learning rates at 200 epochs | e6 | 123.93 | 38.86 |
| `d11_full_train_g1_250ep` | g1's recipe at 250 epochs on all of train | g1 | 123.93 | 38.86 |
| `d12_full_train_g1` | final model: g1's recipe (200 epochs) on all of train | g1 | 123.93 | 38.86 |
| `h1_vit_adapter` | P2 stem replaced by a lite ViT-Adapter (spatial prior injected into / extracted from the ViT) | f5 | 124.86 | 39.79 |
| `h2_patch8` | ViT re-tokenized at 8 px (PI-resized patch embedding: 64 x 64 tokens) | f5 | 122.16 | 37.09 |
| `h3_nwd` | anchors and proposals labelled by max(IoU, NWD) (tiny-object label assignment) | f5 | 123.93 | 38.86 |
| `h4_iou_aware` | IoU branch on the RoI head; detections ranked by class score x predicted IoU^0.5 | f5 | 123.94 | 38.87 |
| `h5_lora_plus` | LoRA+: DoRA B matrices at 8x the adapter learning rate | f5 | 123.93 | 38.86 |
| `h6_p2_heatmap` | training-only CenterNet heatmap loss on P2 | f5 | 124.08 | 39.01 |
| `i1_cls_loss025_200ep` | classification loss weight 1.0 -> 0.25 (trade macro-F1 for mAP) | g1 | 123.93 | 38.86 |
| `i2_cls_detach_200ep` | no classification gradient into the backbone (classifier on detached tokens) | g1 | 123.93 | 38.86 |
| `d13_full_train_i2_250ep` | final candidate: i2's recipe at 250 epochs on all of train | i2 | 123.93 | 38.86 |
| `d14_full_train_i1_cls01_250ep` | final candidate: i1 with classification weight 0.1, 250 epochs, all of train | i1 | 123.93 | 38.86 |
| `j0_base` | new base: f5 + classification 0.1, no mosaic, same-class copy-paste with a 1 px margin | f5 | 123.93 | 38.86 |
| `j1_roi_fused` | every RoI pooled from P2-P5, fused by max (instead of one level) | j0 | 123.93 | 38.86 |
| `j2_roi_canonical56` | RoI level mapping canonical scale 224 -> 56 px | j0 | 123.93 | 38.86 |
| `j3_cls_token` | image classifier on [CLS, mean patch token] | j0 | 123.94 | 38.87 |
| `d15_full_train_j0_250ep` | final candidate: d14 with j0's augmentation | d14 | 123.93 | 38.86 |

The `b*` runs answer "how should the remaining trainable budget be spent": all build on `a1` and fill ~38-39.5 M
of the 40 M a different way. `train.py` refuses to train a model over 130 M total or 40 M trainable.

- Accept the (gated) DINOv3 licence on the model page and `huggingface-cli login` once first (~0.35 GB download).
- Memory and time: [docs/DETAILS.md](docs/DETAILS.md#time-and-memory).

### 2.3 Use Course Data

TA Zichen ran this code on fully labelled data in another folder layout. For the
[course data](https://huggingface.co/datasets/doem1997/cs701-sar-course-data), adapt `sarbench/data.py` and `train.py`:

- train on `train/labels.csv` + `train/instances.json`
- hold out part of train for local validation (val and test have no labels)
- predict on the images in `val/images.json` and `test/images.json`

### 2.4 Make Submission

`make_submission.py` has `write_submission` (the function in the briefing video): maps your boxes back to
original image pixels, writes `submission.zip` for Codabench.

A run also writes the zips itself: `runs/<name>/submission_val.zip` from training (and `submission_test.zip`
when a labelled test split exists), plus `submission_val.zip` / `submission_test.zip` from `--load` (below).
For the unlabelled val/test splits (`val/images.json`, `test/images.json`), which have no labels, predict with
the trained weights and write the zip directly:

```bash
python train.py --config configs/dinov3_dora_giou.yaml \
  --load runs/dinov3_dora_giou/model.pt \
  --predict /path/to/test/images.json --predict-split test --out runs/predict
# -> runs/predict/dinov3_dora_giou/submission_test.zip
```

`--load` alone (without `--predict`) evaluates a saved model on the labelled split and writes its submission
zips; `--predict` requires `--load` and skips metrics, so it needs no labels.

**Post-processing.** `sweep_postprocess.py` (same flags as `train.py`, plus `--score-thresh` / `--nms-thresh` lists)
re-scores a saved model on the held-out split for each score threshold x NMS IoU (x image-class re-scoring, below),
without retraining, and writes `postprocess_sweep.json`. Apply the best setting to `--predict` with `--score-thresh`,
`--nms-thresh`, `--rescore` and `--rescore-power`, or set the matching `model.*` keys in the config.

**Image-class re-scoring** (`--rescore`, off by default). Every image has one class, so a box of any other class is a
false positive. At inference the image classifier's probabilities re-score the boxes before the score threshold, NMS
and top-100: `soft` multiplies each class score by p(class | image)^power, `hard` keeps only the predicted image
class, `merge` gives that class the box's whole foreground probability. Sweep it on an existing model:
`python sweep_postprocess.py ... --score-thresh 0.001 --nms-thresh 0.6 --rescore off soft hard merge --rescore-power 0.5 1 2`.


## 3. Baseline Results (for reference)

Test split, %, mean ± std over 3 seeds. Δm reference: ViT + LoRA, mean of its 3 seeds (93.99 / 25.41), so its Δm
is 0. Full fine-tuning and from scratch train 106 M parameters with the heads: over the 40 M budget, shown for
reference (full fine-tuning: the upper bound).

| backbone | adaptation | trained backbone params | macro-F1 | mAP | Δm |
|---|---|---:|---:|---:|---:|
| ViT (ImageNet-21k) | full fine-tuning | 86.0 M | 95.8 ± 0.3 | 31.8 ± 0.2 | +13.5 ± 0.4 |
| ViT (ImageNet-21k) | LoRA | 0.88 M | 94.0 ± 0.2 | 25.4 ± 0.5 | 0.0 ± 0.9 |
| ViT (ImageNet-21k) | MoE-LoRA | 1.03 M | 93.4 ± 0.5 | 26.8 ± 0.3 | +2.5 ± 0.8 |
| TerraMind (Earth observation) | full fine-tuning | 85.3 M | 95.6 ± 0.2 | 30.8 ± 0.4 | +11.5 ± 0.7 |
| TerraMind (Earth observation) | LoRA | 0.88 M | 92.6 ± 0.4 | 26.5 ± 0.4 | +1.4 ± 1.0 |
| TerraMind (Earth observation) | MoE-LoRA | 1.03 M | 92.4 ± 0.1 | 27.8 ± 0.3 | +3.8 ± 0.7 |
| ViT, from scratch | full training | 86.0 M | 82.8 ± 0.7 | 17.0 ± 0.3 | −22.5 ± 0.9 |
| TerraMind, from scratch | full training | 85.3 M | 86.1 ± 0.2 | 17.8 ± 0.5 | −19.3 ± 0.8 |

<details>
<summary>Val results</summary>

Val split, %, mean ± std over 3 seeds. Δm reference: the same configuration on val (93.31 / 26.13).

| backbone | adaptation | trained backbone params | macro-F1 | mAP | Δm |
|---|---|---:|---:|---:|---:|
| ViT (ImageNet-21k) | full fine-tuning | 86.0 M | 94.4 ± 0.3 | 32.0 ± 0.4 | +11.8 ± 0.7 |
| ViT (ImageNet-21k) | LoRA | 0.88 M | 93.3 ± 0.4 | 26.1 ± 0.4 | 0.0 ± 0.9 |
| ViT (ImageNet-21k) | MoE-LoRA | 1.03 M | 92.2 ± 0.8 | 27.1 ± 0.2 | +1.3 ± 0.9 |
| TerraMind (Earth observation) | full fine-tuning | 85.3 M | 94.2 ± 0.3 | 31.5 ± 0.1 | +10.7 ± 0.3 |
| TerraMind (Earth observation) | LoRA | 0.88 M | 92.2 ± 0.4 | 27.0 ± 0.5 | +1.0 ± 1.3 |
| TerraMind (Earth observation) | MoE-LoRA | 1.03 M | 91.9 ± 0.4 | 28.8 ± 0.1 | +4.4 ± 0.3 |
| ViT, from scratch | full training | 86.0 M | 83.5 ± 0.9 | 18.2 ± 0.5 | −20.3 ± 1.3 |
| TerraMind, from scratch | full training | 85.3 M | 86.0 ± 0.9 | 19.2 ± 0.4 | −17.3 ± 1.2 |

</details>

Accuracy and AP50: [docs/DETAILS.md](docs/DETAILS.md#results-in-full).

## 4. Code Structure

```
train.py              entry: train + evaluate one configuration
configs/              the base recipe; configs/ablations/ one-change ablations of it (`base:` inheritance)
run_all.sh            the base and every ablation, in sequence
make_submission.py    write_submission(): predictions -> submission.zip
sarbench/             the code behind train.py
├── config.py         load a YAML run config into train.py arguments
├── data.py           images + boxes at 512 × 512, flips, rotations, scale jitter, mosaic, copy-paste
├── channels.py       pseudo-RGB assembly for the DINOv3 input (amplitude + despeckled + edge)
├── preprocess.py     SAR-BM3D despeckling and percentile clipping (used by channels.py)
├── backbones.py      DINOv3 ViT (Hugging Face transformers)
├── adapters.py       DoRA + per-block capacity: rank overrides, fully trained blocks, Convpass adapters
├── heads.py          RoI heads with a GIoU term: Faster R-CNN and Cascade R-CNN
├── model.py          backbone + class head + ViTDet pyramid (+ P2 stem, WST stem, fusion) + head; build_model
└── metrics.py        accuracy, macro-F1, COCO box AP (incl. AP small / medium / large)
tests/                checks of sarbench/ (python -m pytest tests; SARBENCH_TEST_DINOV3=1 for the DINOv3 ones)
docs/DETAILS.md       model, training protocol, design notes, time and memory
```

More: [docs/DETAILS.md](docs/DETAILS.md).

## 5. Links

- Data: https://huggingface.co/datasets/doem1997/cs701-sar-course-data
- Leaderboard: https://www.codabench.org/competitions/18271
- Questions: TA Zichen, zichen.tian.2023@phdcs.smu.edu.sg
