# Details of the reference code

Back to the [README](../README.md).

## Model

```
image [B,3,512,512] (pseudo-RGB) -> (input adapter) -> DINOv3 ViT-B/16 + DoRA -> tokens [B,1024,768]
  classification: mean over tokens -> LayerNorm -> Linear(768, 9)
  detection:      tokens (+ optional multi-layer fusion) as a [B,768,32,32] map
                  -> SimpleFeaturePyramid (strides 4 to 64, 256 channels)
                  (+ conv stem and/or WST stem adding real stride-4 detail to P2)
                  -> RegionProposalNetwork + RoI head (Faster R-CNN, or Cascade R-CNN), GIoU box term
loss = cross-entropy + the detection losses, each scaled by its config `loss` weight
```

The base recipe is `configs/dinov3_dora_giou.yaml`; every option below is off there unless stated.

**Backbone.** DINOv3 ViT-B/16 (`facebook/dinov3-vitb16-pretrain-lvd1689m`, 86 M, gated on Hugging Face) through
`transformers`. 1 CLS + 4 register tokens are dropped from the output; RoPE, so no position embedding to resample.

**Adaptation (`peft`, `sarbench/adapters.py`).** The backbone is frozen and then, per block:

- DoRA (Liu et al. 2024) on `peft.targets` (`attn`: q/k/v/o_proj, `mlp`: up/down_proj):
  `W' = m * (W + (alpha / r) B A) / ||W + (alpha / r) B A||_c`, with `m` initialized to `||W||_c`, so it is a no-op
  at init. Base: r = 32, alpha = 32 on attention and MLP of all 12 blocks (5.39 M).
- `rank_overrides: {block: rank}` changes the rank of single blocks; alpha / rank is kept, so the update scale is
  unchanged. Per block, DoRA costs 13,824 parameters per unit of rank plus 6,912 for the magnitudes.
- `full_blocks: [...]` trains those blocks entirely instead (7.09 M each, no DoRA there). Unfreezing adds trainable
  parameters but not total ones.
- `conv_adapters: {layers: [...], dim: d}` adds a Convpass adapter (Jie & Deng 2022) after those blocks:
  `patch tokens += up(GELU(conv3x3(GELU(down(tokens)))))`, zero-initialized. DoRA acts on each token separately;
  this is the only option that mixes neighbouring patches inside the backbone (1,536 d + 9 d^2 per block).

**Input adapter** (`model.input_adapter`). DINOv3 was pretrained on ImageNet-normalized RGB; the pseudo-RGB
channels are in [0, 1]. The adapter applies the ImageNet mean/std, then a learnable 1x1 conv initialized to the
identity, and unfreezes the patch embedding (0.59 M) so the first layer can learn the SAR channels.

**Learning rates.** Three AdamW groups: the heads and every new module outside the backbone at `train.lr` (1e-4),
new backbone parameters (DoRA, Convpass) at `train.backbone_lr` (default `lr`), and pretrained weights made
trainable (patch embedding, `full_blocks`) at `train.unfrozen_lr` (2e-5, the course's full fine-tuning rate).

**Budget.** `train.py` logs total / trainable parameters (with the heads / new backbone / unfrozen split) and
refuses to train over 130 M total or 40 M trainable. `tests/test_config.py` checks every config.

**P2 stem** (`model.detail_stem`, on in the base). The ViT sees only 16 px patches, so the neck's stride-4 level is
upsampled. A small conv stem (two stride-2 convs on the 512 px input, ~0.1 M) adds real stride-4 detail to P2.

**Convpass variants** (`peft.conv_adapters`): `kernel` dense (one 3x3 conv on the `dim` channels, the default),
depthwise (one 3x3 per channel, so `dim` can be much wider for the same parameters) or multiscale (two 3x3 convs of
`dim` / 2 output channels side by side, dilation 1 and 2: 3x3 and 5x5 token context for the dense conv's parameters);
`placement` sequential (after the block, the default) or parallel (`y = block(x) + adapter(LayerNorm(x))`, beside
attention and MLP, as in the Convpass paper). Every variant starts as a no-op.

**Box head** (`model.head_params.box_head`, both heads): `mlp` (torchvision's two fc layers of width `hidden_dim`;
the first, 256*7*7 -> 1024, is 12.85 M) or `conv` (`ConvFCBoxHead`: `num_convs` 3x3 convs with GroupNorm on the 7x7
RoI, a 1x1 down to `reduce` channels, one fc of width `hidden_dim`: 5.6 M at 4 / 256 / 64 / 1024).

**RPN and post-processing** (`model.*`): `rpn_fg_iou_thresh` / `rpn_bg_iou_thresh` (0.7 / 0.3) label the anchors;
every GT keeps its best anchor regardless. `score_thresh` / `nms_thresh` / `detections_per_img` (0.05 / 0.5 / 100)
only affect inference; `sweep_postprocess.py` tunes them on a saved model, `--score-thresh` / `--nms-thresh` override
them for `--predict`. COCO AP counts at most 100 detections per image.

**P2 stem size** (`model.detail_stem_params`): `channels` (64; the second conv has twice as many) and `extra_convs`
(0) more 3x3 convs at stride 4.

**EMA** (`train.ema_decay`, 0 = off). An exponential moving average of the trainable weights, updated after every
optimizer step (decay ramped from ~0.1 to `ema_decay`), is used for validation, the final test and `model.pt`.

**Full train** (`data.val_fraction: 0`). Trains on all of train with no held-out validation; the run writes
`model.pt` and `metrics.json` only, and the zips come from `--load ... --predict`.

**WST stem** (`model.wst_stem`). MSFA's best filter: a wavelet scattering transform (kymatio, J = 2, L = 8, 81
channels at stride 4, P2's resolution) on the first input channel, projected to 256 channels and added to P2
(~0.6 M).

**Multi-layer fusion** (`model.fusion_layers: [5, 8, 11]`). Forward hooks capture those blocks' tokens; a learnable
softmax weighting (initialized to the last block) combines them before the neck. Cost: one weight per block.

**RoI head** (`model.head`). `standard`: torchvision's 2-layer MLP box head (hidden 1024, 13.9 M). `cascade`:
Cascade R-CNN, S stages (`head_params.num_stages`, default 3) with rising IoU thresholds
(`fg_iou_thresholds`), per-stage or shared box heads (`share_head`, `hidden_dim`) and `stage_loss_weights`; at
inference intermediate stages pass every refined proposal on, so small-object recall is kept. Earlier Cascade
runs lost mAP to the standard head, but none isolated it on warm-started heads, which is what
`a6_warmstart_cascade` tests (against `a4_warmstart`).

**GIoU** (`model.giou_weight`, 1.0 in the base): an extra GIoU term on the decoded positive boxes vs their matched GT,
added to the RoI box loss (localization / AP75). `model.roi_sampling_ratio` / `roi_output_size` (2 / 7) set the
RoIAlign. The RPN keeps the top 2000 proposals after NMS in training and testing.

**Loss weights** (`loss`). The epoch loss is the sum of the terms, each multiplied by its config weight (default
1.0): `classification`, `loss_objectness`, `loss_rpn_box_reg`, `loss_classifier`, `loss_box_reg`. The base uses
`loss_box_reg: 2.0` and `loss_rpn_box_reg: 1.5`. The logged per-term values are raw.

**Augmentation** (`data.*`, train split only, all after the resize to 512 px; the pseudo-RGB channels are computed
and cached before any augmentation):

- flips (always), `rotation90` (an exact `torch.rot90` of the pixels and boxes),
- `scale_jitter: [lo, hi]` — shrinking pads the border (nothing is lost), enlarging crops the window that keeps the
  most boxes and drops boxes left with less than half of their area,
- `mosaic` (probability; four images at 256 px in a 2x2 grid) and `copy_paste` (probability; up to 3 donor objects
  pasted where they overlap nothing, IoU < 0.3) with `copy_paste_margin` px of context around each object.

The base uses `mosaic: 0.3`, `copy_paste: 0.5`, no rotation, jitter or margin.

**Pseudo-RGB input (`preprocess.pseudo_rgb`, on in the base).** `sarbench/channels.py` builds three [0, 1] channels:
`mode: despeckle` (base) = amplitude (`sigma_clip`), SAR-BM3D despeckled base, Sobel edge of the base;
`amp_log_edge` = amplitude, log-amplitude, edge of the amplitude; `amplitude` = amplitude three times.
`preprocess.cache` caches the result (the SAR-BM3D pass is slow).

**Two-stage training** (`train.stage_a_epochs`, `stage_a_lr`). The first N epochs freeze the whole backbone and
train only the heads at `stage_a_lr`, so the random heads do not push noisy gradients into DoRA (MSFA's "model
gap"); the usual `epochs` follow with a fresh optimizer and schedule.

## Training protocol

- AdamW, weight decay 0.05, batch size 16, bf16 autocast, 48 epochs (base).
- One epoch of linear warmup, then cosine decay to 0, updated every iteration.
- Validation every 4 epochs and after the last one; test once, at the end. The reported model is the
  last-epoch model: there is no checkpoint selection.

## Design notes

- **Both labels on every image**, so every training batch trains both heads.
- **512 px input.** Every image is resized to 512 x 512 (non-square images are stretched): 1,024 tokens.
  Objects are small (median box 15.9 px at 512 px), so the smallest anchors are 8 px, on the stride-4 level.
  Predicted boxes are mapped back to original pixels before COCO scoring.
- **Normalization.** Without pseudo-RGB, the gray image is normalized with the train mean and std (0.2526, 0.2133)
  and repeated to RGB. With pseudo-RGB the three [0, 1] channels go in directly, or through the input adapter.
- **Precision rule.** Backbone, neck and classification head run in bf16; the RPN and RoI heads run in fp32.
  torchvision's box coder casts anchors to the dtype of the regression output, and bf16 numbers between 256
  and 512 are 2 px apart: too coarse for objects of 6-15 px. `train.py` enables TF32 for fp32 matmuls and
  convolutions (importing terratorch used to do this implicitly); box coordinates are decoded elementwise in
  fp32 and are unaffected.

## Time and memory

One DINOv3 run takes roughly 1.5-2 h on one L40S (about 50 min on an RTX PRO 6000 Blackwell). A run keeps about
11-17 GiB of GPU memory in use; the capacity ablations add optimizer state for up to 14 M more parameters, and
the Convpass and high-rank DoRA runs add activations. A smaller `--batch-size` saves memory but changes the
protocol.

## What a run writes

`runs/<name>/`, where `<name>` is the config's `name` (default `dinov3_dora`):

- `log.txt`: training progress and the val and test scores.
- `metrics.json`: arguments, parameter counts (and their breakdown), `train_hours` (wall time of the epoch loop, including the
  periodic validation; it grows if the GPU is shared), `peak_gpu_memory_gb` (`torch.cuda.max_memory_allocated`
  in GiB; the process needs about 2.5 GiB more), every validation result, and the final val and test metrics.
- `predictions_val.json`, `predictions_test.json`: detections in COCO result format, in original pixels.
- `model.pt`: the final weights (`state_dict`).

## Checks

```bash
bash run_all.sh --limit 64 --epochs 1 --out runs/smoke   # every configuration on 64 images per split
python -m pytest tests                                   # CPU tests; GPU and dataset tests where available
SARBENCH_TEST_DINOV3=1 python -m pytest tests            # + the real DINOv3 checks (no-op adapters, budgets)
```

`test_data.py`, `test_metrics.py` and `test_model.py` read the data in TA Zichen's layout (see the README) and
need a GPU; the other tests run on the CPU without data.

## Results in full (course reference runs)

The course's reference runs (ViT and TerraMind; their code is no longer in this repository). They define the
Δm reference: ViT-B/16 + LoRA.

Scores in %, mean ± sample standard deviation over seeds 0, 1 and 2, last-epoch model. Accuracy and macro-F1
for classification; COCO mAP (AP@[.50:.95]) and AP50 for detection. The seeds differ by a few tenths of a
point, so smaller differences are noise. Δm is computed per seed against the reference, the mean of the three
ViT-B/16 + LoRA runs (test 93.99 / 25.41, val 93.31 / 26.13; the Codabench leaderboard uses the same values), then
averaged, so the reference configuration shows 0. Full fine-tuning and training from scratch train 106 M
parameters with the heads, over the 40 M trainable budget of the assignment: they are references, full fine-tuning
the upper bound.

Test (2,065 images):

| backbone | init | adapt | trainable backbone params (M) | accuracy | macro-F1 | mAP | AP50 | Δm |
|---|---|---|---:|---:|---:|---:|---:|---:|
| vit | pretrained | full | 86.0 | 95.9 ± 0.3 | 95.8 ± 0.3 | 31.8 ± 0.2 | 62.4 ± 0.1 | +13.5 ± 0.4 |
| vit | pretrained | lora | 0.88 | 94.2 ± 0.2 | 94.0 ± 0.2 | 25.4 ± 0.5 | 53.9 ± 0.4 | 0.0 ± 0.9 |
| vit | pretrained | moelora | 1.03 | 93.7 ± 0.5 | 93.4 ± 0.5 | 26.8 ± 0.3 | 55.9 ± 0.5 | +2.5 ± 0.8 |
| terramind | pretrained | full | 85.3 | 95.8 ± 0.2 | 95.6 ± 0.2 | 30.8 ± 0.4 | 61.8 ± 0.4 | +11.5 ± 0.7 |
| terramind | pretrained | lora | 0.88 | 93.1 ± 0.3 | 92.6 ± 0.4 | 26.5 ± 0.4 | 55.4 ± 1.2 | +1.4 ± 1.0 |
| terramind | pretrained | moelora | 1.03 | 92.9 ± 0.1 | 92.4 ± 0.1 | 27.8 ± 0.3 | 57.7 ± 0.4 | +3.8 ± 0.7 |
| vit | scratch | full | 86.0 | 83.9 ± 0.6 | 82.8 ± 0.7 | 17.0 ± 0.3 | 36.7 ± 0.4 | −22.5 ± 0.9 |
| terramind | scratch | full | 85.3 | 86.7 ± 0.2 | 86.1 ± 0.2 | 17.8 ± 0.5 | 38.9 ± 0.4 | −19.3 ± 0.8 |

Val (1,426 images):

| backbone | init | adapt | trainable backbone params (M) | accuracy | macro-F1 | mAP | AP50 | Δm |
|---|---|---|---:|---:|---:|---:|---:|---:|
| vit | pretrained | full | 86.0 | 94.7 ± 0.3 | 94.4 ± 0.3 | 32.0 ± 0.4 | 62.7 ± 0.4 | +11.8 ± 0.7 |
| vit | pretrained | lora | 0.88 | 93.6 ± 0.5 | 93.3 ± 0.4 | 26.1 ± 0.4 | 54.7 ± 0.6 | 0.0 ± 0.9 |
| vit | pretrained | moelora | 1.03 | 92.6 ± 0.8 | 92.2 ± 0.8 | 27.1 ± 0.2 | 56.4 ± 0.3 | +1.3 ± 0.9 |
| terramind | pretrained | full | 85.3 | 94.5 ± 0.2 | 94.2 ± 0.3 | 31.5 ± 0.1 | 62.4 ± 0.0 | +10.7 ± 0.3 |
| terramind | pretrained | lora | 0.88 | 92.6 ± 0.4 | 92.2 ± 0.4 | 27.0 ± 0.5 | 55.9 ± 0.8 | +1.0 ± 1.3 |
| terramind | pretrained | moelora | 1.03 | 92.2 ± 0.5 | 91.9 ± 0.4 | 28.8 ± 0.1 | 58.7 ± 0.1 | +4.4 ± 0.3 |
| vit | scratch | full | 86.0 | 83.8 ± 0.8 | 83.5 ± 0.9 | 18.2 ± 0.5 | 39.0 ± 0.7 | −20.3 ± 1.3 |
| terramind | scratch | full | 85.3 | 86.0 ± 1.0 | 86.0 ± 0.9 | 19.2 ± 0.4 | 41.4 ± 0.4 | −17.3 ± 1.2 |
