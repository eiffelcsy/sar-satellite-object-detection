# Results

Back to the [README](../README.md). Every training run is in [results/wandb_runs.csv](results/wandb_runs.csv)
(80 runs, pulled from W&B: parent run, the one change, held-out metrics, parameters, hours) and the RF-DETR runs in
[results/wandb_rfdetr_runs.csv](results/wandb_rfdetr_runs.csv).

**Δm** (the leaderboard score) is the mean relative change of macro-F1 and mAP against the course reference,
ViT-B/16 + LoRA: `Δm = 100 × ½ [(F1 − F1_ref) / F1_ref + (mAP − mAP_ref) / mAP_ref]`, with F1_ref / mAP_ref =
93.31 / 26.13 on val and 93.99 / 25.41 on test. One mAP point is worth ~1.9 Δm, one F1 point ~0.54.

## Final model (d16)

| split | Δm |
|---|---:|
| Codabench val (1,426 images) | **+44.50** |
| Codabench test (2,065 images) | **+48.20** |

For comparison, the course's full fine-tuning baseline (over the 40 M budget) scores +13.5 on test. One training
seed; 123.93 M parameters in total, 38.86 M trainable; 19.6 h on one NVIDIA L40S, 19.7 GB peak memory.
The submissions use score threshold 1e-5, at most 100 boxes per image and NMS IoU 0.8 (val) / 0.85 (test);
`scripts/predict.sh` reproduces both and scores them against the labels released on 13 Nov 2026 (val: macro-F1
94.86, mAP 48.95).

## Codabench val: from Δm 19.16 to 44.50

Each row adds one change to the row above; one seed per run.

| run | change | val Δm |
|---|---|---:|
| `b4_convpass_58ep` | DINOv3 + DoRA + Convpass, 58 epochs | 19.16 |
| `b4_convpass_72ep` | 72 epochs | 21.80 |
| `d1_96ep` | 96 epochs | 26.65 |
| `e1_stem_96ep` | wider P2 stem (128 channels + one 3x3 conv) | 28.33 |
| `e4_stem_120ep` | 120 epochs | 30.51 |
| `e5_stem_150ep` | 150 epochs | 32.55 |
| `e6_stem_200ep` | 200 epochs | 35.39 |
| `g1_split_lr_200ep` | split learning rates (adapters 3e-4, heads 2e-4, classifier 1e-4) | 36.19 |
| `d10_full_train_e6` | e6 on all of train (no held-out split) | 38.50 |
| `d14_full_train_i1_cls01_250ep` | classification loss weight 0.1, 250 epochs, all of train | 43.58 |
| **`d16_full_train_j1_250ep`** | no mosaic + same-class copy-paste, fused RoI pooling | **44.50** |

## Held-out split: mAP 24.9 → 46.3

939 images (10 %) held out from train, one seed, last epoch, post-processing at score 0.05 / NMS 0.5.

| run | change | epochs | mAP | AP75 | macro-F1 |
|---|---|---:|---:|---:|---:|
| `dinov3_lora` | DINOv3 + pseudo-RGB + LoRA | 24 | 24.86 | 18.92 | 95.39 |
| `dinov3_dora` | DoRA r=32 + P2 stem, 48 epochs | 48 | 32.15 | 29.72 | 95.21 |
| `dinov3_dora_giou` | + GIoU box term | 48 | 32.28 | 30.99 | 95.69 |
| `a1_input` | + input adapter | 48 | 32.21 | 30.11 | 94.79 |
| `b4_convpass` | + Convpass (256) after every block | 48 | 33.89 | 32.58 | 96.19 |
| `d1_96ep` | 96 epochs | 96 | 39.63 | 40.32 | 96.49 |
| `e1_stem_96ep` | wider P2 stem | 96 | 39.95 | 40.36 | 96.89 |
| `f5_split_lr` | split learning rates | 96 | 41.15 | 41.88 | 96.54 |
| `j0_base` | classification weight 0.1, mosaic off, same-class copy-paste | 96 | 45.53 | 47.29 | 96.32 |
| **`j1_roi_fused`** | fused RoI pooling (P2-P5, max): **the d16 recipe at 96 epochs** | 96 | **46.34** | **49.03** | 96.28 |

Seed noise: `e3_d1_seed1` (d1 with seed 1) differs from d1 by 0.05 mAP and 0.47 AP_small, so gaps of about 0.1
mAP are noise.

## What else was tried (held-out Δ mAP vs the parent run)

| idea | run | Δ mAP |
|---|---|---:|
| Convpass vs more DoRA rank at equal budget (~39 M) | `b4` vs `b3_rank_uniform` / `b2_rank_early` / `b1_full_blocks` | +1.68 vs +1.36 / +0.29 / +0.53 |
| DoRA r64 + Convpass 192 (best, never carried forward) | `c2_rank64_convpass192` | +2.02 |
| longer schedules | `b4` 48 → 72 → 96 → 120 epochs | +3.81, +1.93, +1.69 |
| classification weight 0.25 / stop-gradient (200 epochs) | `i1` / `i2` | +2.36 (F1 +0.34) / +1.99 (F1 −1.70) |
| RoI level canonical scale 56 / classifier on [CLS, mean] | `j2` / `j3` | +0.40 / +0.20 |
| Cascade R-CNN head (three variants) | `dinov3_dora_cascade`, `a6`, `c1b` | −0.53, −0.90, −0.52 |
| EMA of the weights | `d2_ema` | −0.54 |
| ViT-Adapter instead of the P2 stem | `h1_vit_adapter` | −0.97 |
| 8 px tokens (3.5x the compute) | `h2_patch8` | −0.98 |
| head-only warm start, at equal compute | `c1a` vs `b4_58ep` | −1.37 |
| narrower / conv box head to fund wider Convpass | `d3` / `d4` | −1.97 / −1.23 |
| NWD label assignment | `h3_nwd` | −3.18 |
| rotations + scale jitter (under mosaic) | `a2_aug` | −5.19 |
| adapter learning rate 1e-3 | `f2_adapter_lr1e-3` | −5.35 |
| on j1: IoU-aware scoring, GIoU x2, box loss x4, alpha-GIoU, no copy-paste | `k3`, `k4`, `k5`, `k7`, `k6` | −0.22, −0.17, −0.69, −0.67, −0.28 |
| on j1: size-aware RPN, centred anchors, hard-class oversampling | `k14`, `k15`, `k16` | −1.30, −0.65, −1.02 |
| stronger copy-paste for small classes | `k1` | stopped: macro-F1 fell to 0.66 |

## Re-runs after the presentation

Three clean one-change ablations of j1 (the d16 recipe at 96 epochs on the held-out split), run on 12-13 Nov 2026.

| run | change vs j1 | epoch | mAP | AP75 | AP_small | macro-F1 | Δ mAP |
|---|---|---:|---:|---:|---:|---:|---:|
| `j1_roi_fused` | (reference) | 96 | 46.34 | 49.03 | 40.72 | 96.28 | |
| `l1_input_amplitude` | pseudo-RGB → amplitude repeated 3x (no SAR-BM3D, no edges) | 96 | 46.87 | 48.17 | 40.47 | 96.33 | +0.53 |
| `l2_j1_no_rope_aug` | DINOv3's training-mode RoPE rescale off | 96 | 46.60 | 49.50 | 40.85 | 95.89 | +0.26 |
| `l3_j1_cascade` | Cascade R-CNN (3 x 384, IoU 0.5/0.6/0.7, scores averaged) | 88 | 45.95 | 48.54 | 40.02 | 96.12 | −0.36 |

- **l1** is the isolating run the pseudo-RGB input never had: the SAR-BM3D and edge channels did not help on the
  held-out split (+0.53 mAP without them, −0.86 AP75; one seed). The final model keeps the pseudo-RGB input it was
  trained with.
- **l2**: no clear effect (+0.26 mAP, −0.39 F1).
- **l3** crashed at epoch 91; at its last validation (epoch 88) it trails j1 at epoch 88 (46.31) by 0.36 mAP, in
  line with the earlier cascade runs. 39.87 M trainable.

## RF-DETR track

A detection-pretrained RF-DETR trained entirely, on the same held-out split (its code is no longer in the
repository). Its best run matched j1's held-out mAP; it was not used for the final model.

| run | epochs | mAP | AP75 |
|---|---:|---:|---:|
| RF-DETR Small, raw input | 79 | 44.82 | 47.93 |
| RF-DETR Large, amplitude/log/edge input, bridge + playground repeated | 199 | 46.65 | 49.39 |
| + frozen DINOv3 ViT-B fused in | 199 | 46.81 | 49.81 |
| + DoRA on the DINOv3 + stride-8 level | 199 | 46.58 | 49.52 |

## Limitations

- About two-thirds of the val and test images have a near-copy in train (flipped, rotated, shifted or cropped
  tiles of the same scene, per the dataset card), so val and test scores overstate generalisation; the held-out
  split shares this.
- One seed per run, and 8 of the 19 changes were bundled with others: the backbone swap, DoRA and the P2 stem never
  had an isolating run (pseudo-RGB got one only with l1), so they are design choices, not ablations.
- Bridge AP stayed at 4-6 through every change (40 % of bridges are never proposed by the RPN); AP_small 41 vs
  AP_large 67 on the held-out split.
- Mosaic halved every object (median 15.9 → 8 px) and mislabelled three quarters of each mosaic in 51 of 77 runs
  before it was caught; removing it (`j0`) was the largest held-out step.

## Course reference baselines (test, mean ± std of 3 seeds)

| backbone | adaptation | trainable backbone params | macro-F1 | mAP | Δm |
|---|---|---:|---:|---:|---:|
| ViT-B/16 (ImageNet-21k) | full fine-tuning (over budget) | 86.0 M | 95.8 ± 0.3 | 31.8 ± 0.2 | +13.5 ± 0.4 |
| ViT-B/16 (ImageNet-21k) | LoRA (the Δm reference) | 0.88 M | 94.0 ± 0.2 | 25.4 ± 0.5 | 0.0 ± 0.9 |
| ViT-B/16 (ImageNet-21k) | MoE-LoRA | 1.03 M | 93.4 ± 0.5 | 26.8 ± 0.3 | +2.5 ± 0.8 |
| TerraMind | full fine-tuning (over budget) | 85.3 M | 95.6 ± 0.2 | 30.8 ± 0.4 | +11.5 ± 0.7 |
| TerraMind | LoRA | 0.88 M | 92.6 ± 0.4 | 26.5 ± 0.4 | +1.4 ± 1.0 |
