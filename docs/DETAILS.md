# Details of the reference code

Back to the [README](../README.md).

## Model

```
image [B,C,512,512] -> backbone -> tokens [B,1024,768]   (32 x 32 patches of 16 px; C=1 or 3)
  classification: mean over tokens -> LayerNorm -> Linear(768, 9)
  detection:      tokens as a [B,768,32,32] map -> SimpleFeaturePyramid (strides 4 to 64, 256 channels)
                  -> detector: RegionProposalNetwork + RoIHeads (torchvision Faster R-CNN) by default,
                     or a Deformable-DETR head (query-based, no anchors); 9 classes + background
loss = cross-entropy + the detection losses
```

![The reference model](reference_model.png)

**Detector** (set in the config's `detector.name`):

- `faster_rcnn` (default): torchvision Faster R-CNN (RPN + RoI heads), ~20 M with the neck.
- `deformable_detr`: a compact Deformable-DETR head (`sarbench/detr.py`, Zhu et al. 2021), ~11 M at hidden
  256, 6 encoder + 6 decoder layers, 300 queries. It projects every pyramid level to 256, runs multi-scale
  deformable attention, and predicts boxes directly (no anchors/NMS). Training uses Hungarian matching + CE
  (no-object class) + L1 + GIoU; evaluation returns the same `{'boxes','labels','scores'}` format. Its
  `level_names` picks which neck levels feed it (`'0'`=stride 4 … `'pool'`=stride 64);
  `configs/dinov3_deformable_detr_lora.yaml` uses P2–P5. The head width is fixed at 256 regardless of the
  backbone, keeping it inside the parameter budget.

**Loss weights** (`loss` in the config). The epoch loss is the sum of the terms below, each multiplied by its
config weight (default 1.0), applied in `train.py`. The per-term values logged in parentheses are always raw.
Use the key names of the active detector:

- always: `classification` (the multi-task class head, cross-entropy).
- `deformable_detr`: `loss_classifier` (query set CE), `loss_bbox` (L1), `loss_giou`, `loss_aux` (auxiliary
  decoder layers). E.g. `loss: {loss_bbox: 1.5, loss_giou: 1.5}` leans on the boxes.
- `faster_rcnn`: `loss_objectness`, `loss_rpn_box_reg`, `loss_classifier`, `loss_box_reg` (`loss_box_reg` is
  the RoI box regression; `loss_rpn_box_reg` the RPN one).

For `deformable_detr`, the Hungarian matcher costs (`cost_bbox`, `cost_giou`) and the no-object weight
(`eos_coef`) live in `sarbench/detr.py` and affect assignment, not the summed loss; pass them via
`detector.params` if needed.

**Backbones** (set in the config's `backbone.name`):

- `vit`: timm's `vit_base_patch16_224.augreg_in21k` (ImageNet-21k).
- `terramind`: TerraMind-1.0-base (IBM/ESA), an Earth-observation model, through its Sentinel-1 GRD input.
- `dinov3`: DINOv3 ViT (Meta) through HuggingFace transformers; the default checkpoint is
  `facebook/dinov3-vitb16-pretrain-lvd1689m` (~86 M). The configuration is set in `configs/dinov3_*.yaml`;
  `model_name` can be swapped for `facebook/dinov3-vits16-pretrain-lvd1689m` (lighter) or
  `facebook/dinov3-vitl16-pretrain-sat493m` (satellite, ~303 M, over the parameter budget). DINOv3 weights
  are gated: accept the licence and `huggingface-cli login`.
  Adapting DINOv3 uses its separate `q_proj` / `k_proj` / `v_proj` (the "qkv_proj") and `o_proj` ("out_proj")
  attention layers plus the `up_proj` / `down_proj` MLP layers (fc1/fc2), selected by the config's
  `peft.targets: [attn, mlp]`. `configs/dinov3_lora.yaml` uses r = 32, alpha = 32.

**Pseudo-RGB input (`preprocess.pseudo_rgb`, DINOv3 configs).** Instead of repeating the gray channel to the
three channels DINOv3 expects, `sarbench/channels.py` assembles three complementary [0, 1] representations:

1. normalized amplitude — the intensity image with its dynamic range stretched (`sigma_clip`);
2. despeckled base — SAR-BM3D, or a Gaussian blur when `despeckle: false`;
3. edge map — the Sobel gradient magnitude of the base (or a Gaussian high-pass with `edge: highpass`).

The assembly is an ordinary pre-processing callable, so `--preprocess-cache` caches the SAR-BM3D pass across
epochs; `configs/dinov3_*.yaml` set `cache: preprocess_cache`. Adding `pseudo_rgb` with a non-3-channel
backbone (e.g. the ViT) is rejected at startup. With `pseudo_rgb: false` DINOv3 falls back to repeating the
single gray channel to RGB.

**Adaptation** (set in the config's `peft.method`):

- `full`: every backbone weight trains.
- `lora`: the backbone is frozen; the selected projections get `y = W x + (alpha / r) B A x`, with r = 16 and
  alpha = 32 for ViT/TerraMind (0.88 M trainable backbone parameters for the ViT).
- `moelora`: as `lora`, but the rank-16 update is split into 4 experts of rank 4 (alpha = 8, so alpha / r = 2
  as in `lora`), mixed per token by a softmax router. Each task has its own router (1.03 M trainable backbone
  parameters for the ViT).
- `peft.targets` (optional): which module groups a backbone exposes for adaptation. ViT/TerraMind have only
  `attn` (their `qkv`/`proj`); DINOv3 supports `attn` (q/k/v/o_proj) and `mlp` (up/down_proj, fc1/fc2).
  `configs/dinov3_lora.yaml` sets `[attn, mlp]`; without it, only attention is adapted.

Both registries are plain dicts (`sarbench/backbones.py`, `sarbench/adapters.py`): add a new backbone with
`@register_backbone('name')` (exposing `embed_dim`, `blocks`, `adapter_targets()` and a token forward) or a
new PEFT method with `@register_adapter('name')`, then select it from a YAML config. `adapter_targets()` makes
the adapters independent of the attention layout (fused qkv vs. separate q/k/v/o).

Every run also trains the 20.1 M parameters of the heads (neck, RPN, RoI heads, classifier), so
`trainable_parameters` in `metrics.json` is the backbone count + 20.1 M.

## Training protocol (the same for all eight runs)

- AdamW, weight decay 0.05, batch size 16, bf16 autocast, 24 epochs.
- Learning rate 1e-4; the backbone of the two pretrained `full` runs uses 2e-5 (`--backbone-lr 2e-5`, as in
  `run_all.sh`), so a pretrained backbone is fine-tuned gently.
- One epoch of linear warmup, then cosine decay to 0, updated every iteration.
- Validation every 4 epochs and after the last one; test once, at the end. The reported model is the
  last-epoch model: there is no checkpoint selection.

## Design notes

- **Both labels on every image**, so every training batch trains both heads.
- **512 px input.** Every image is resized to 512 x 512 (non-square images are stretched): 1,024 tokens.
  Objects are small (median box 15.9 px at 512 px), so the smallest anchors are 8 px, on the stride-4 level.
  Predicted boxes are mapped back to original pixels before COCO scoring.
- **Normalization.** Images are read as one gray channel. Both backbones get the same mean and std
  (0.2526, 0.2133), computed over the 9,392 train images at 512 px. DINOv3 with pseudo-RGB instead receives
  its three [0, 1] channels directly, without this gray-level normalization.
- **Precision rule.** Backbone, neck and classification head run in bf16; the RPN and RoI heads run in fp32.
  torchvision's box coder casts anchors to the dtype of the regression output, and bf16 numbers between 256
  and 512 are 2 px apart: too coarse for objects of 6-15 px. Importing terratorch (`sarbench/backbones.py` does, for
  every run) switches fp32 matrix multiplications on the GPU to TF32; box coordinates are decoded
  elementwise in fp32 and are unaffected.
- **MoE-LoRA runs the backbone twice.** Its routers are task-specific, so the two tasks see different
  features and every forward pass runs the backbone once per task.
- **TerraMind domain gap.** TerraMind was pretrained on Sentinel-1 GRD backscatter in dB (polarizations VV
  and VH, 10 m pixels). Our images are 8-bit gray display images, mostly at a higher resolution; the gray
  channel is fed as both VV and VH. The gap is part of the benchmark and is not corrected in code.

## Time and memory

On one RTX PRO 6000 Blackwell GPU, a run alone takes about 50 min (LoRA about 47 min, MoE-LoRA about 1 h);
all eight in sequence take about 7 h. A run keeps about 11-17 GiB of GPU memory in use; TerraMind with
MoE-LoRA needs the most and does not fit on a 16 GB card. A smaller `--batch-size` saves memory but changes
the protocol.

## What a run writes

`runs/<name>/`, where `<name>` is the config's `name` (default `<backbone>_<init>_<adapt>`):

- `log.txt`: training progress and the val and test scores.
- `metrics.json`: arguments, parameter counts, `train_hours` (wall time of the epoch loop, including the
  periodic validation; it grows if the GPU is shared), `peak_gpu_memory_gb` (`torch.cuda.max_memory_allocated`
  in GiB; the process needs about 2.5 GiB more), every validation result, and the final val and test metrics.
- `predictions_val.json`, `predictions_test.json`: detections in COCO result format, in original pixels.
- `model.pt`: the final weights (`state_dict`).

## Checks

```bash
bash run_all.sh --limit 64 --epochs 1 --out runs/smoke   # all eight configurations on 64 images per split, a few minutes
python -m pytest tests                                   # the tests (need a GPU)
```

`test_data.py`, `test_metrics.py` and `test_model.py` read the data in TA Zichen's layout (see the README),
so they need the same adaptation as the code; `test_backbones.py` and `test_adapters.py` need no data.

## Results in full

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
