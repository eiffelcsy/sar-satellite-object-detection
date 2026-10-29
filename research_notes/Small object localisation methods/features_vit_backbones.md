# Feature representation for small-object detection with plain ViT / foundation-model backbones

Context: frozen DINOv3 ViT-B/16 + DoRA r=32 + Convpass 256, ViTDet SimpleFeaturePyramid + 128-ch P2 conv stem, torchvision Faster R-CNN, 512 px input, objects mostly < 32 px. Val mAP .463 / AP75 .490 / AP_S .407. About 1 M trainable params left (<= 40 M trainable, <= 130 M total). Already negative: ViT-Adapter, block-level fusion, Convpass variants, WST stem, mosaic, EMA, rot90+jitter, warm start, CLS-token classifier.

Research note: about 20 tool calls. Some PDFs (ViTDet, DINOv3 appendix D.9) could only be read in part. Every number below comes from the cited page. Items I could not verify are listed under Gaps.

## Q1. Getting high-resolution detail from a stride-16 ViT (P2 stems, spatial priors, feature upsamplers, ViT-CoMer, DINOv3 recipe, patch size, registers)

### Takeaway
The best-documented gains on COCO come from adding a convolutional spatial-prior branch that is fused with the ViT (ViT-Adapter, ViT-CoMer). These add +4 to +6 AP over a plain ViT/ViTDet at about 5 to 15 M extra parameters, but they were measured with full fine-tuning, and ViT-Adapter has already failed here. Learned feature upsamplers (CARAFE, FADE, DySample) give about +1.5 to 2 AP_S in CNN-FPN detectors for very few parameters, making them the cheapest high-resolution option. Foundation-model upsamplers (LoftUp, JAFAR, FeatUp, LiFT) have no published detection numbers. DINOv3's own detector is not reproducible at this budget: it uses a 7B backbone, a 100 M-parameter decoder and 2048 px inputs.

### Cited Findings
**ViT-CoMer (CVPR 2024, Xia et al., arXiv 2403.07392; code github.com/Traffic-X/ViT-CoMer)**
- Mechanism: a multi-receptive-field CNN branch (MRFP, depth-wise convs) runs in parallel with a plain ViT. CTI modules fuse CNN and transformer features in both directions. No extra pre-training is needed. — [arXiv/HF](https://huggingface.co/papers/2403.07392)
- Mask R-CNN 1x box AP / AP75 on COCO (full fine-tune):
  - ViT-B: 42.9 / 46.8
  - ViTDet-B: 43.2 / 46.9 (121 M params)
  - ViT-Adapter-B: 47.0 / 51.4 (120 M)
  - ViT-CoMer-B: 47.6 / 51.9 (129 M)
  - ViT-CoMer-B with DINOv2 weights: 52.0 / 57.2, and 54.2 / 59.4 at 3x. — [ViT-CoMer HTML](https://arxiv.org/html/2403.07392v3)
- Ablation on ViT-S, 1x:
  - Plain ViT-S: 40.2 AP
  - + MRFP alone: 41.5
  - + CTI (into ViT): 43.3
  - + bidirectional CTI: 45.8
  - Over half the gain comes from feeding ViT features back into the CNN branch. — [ViT-CoMer HTML](https://arxiv.org/html/2403.07392v3)
- Number of interaction modules (N) on ViT-S:
  - N=0: 40.2 AP, 43.8 M params
  - N=2: 45.1 AP (+4.4 M)
  - N=4: 45.8 AP, 50.3 M
  - N=6: 45.6 AP
- Kernel sizes: {3} gives 45.7 and {3,5} gives 45.8, so extra kernels do not help. — [ViT-CoMer HTML](https://arxiv.org/html/2403.07392v3)
- The paper does not report AP_S/M/L. Detection runs use full fine-tuning, not a frozen backbone. — [ViT-CoMer HTML](https://arxiv.org/html/2403.07392v3)

**Feature upsamplers in detection necks (CNN-FPN evidence)**
- Faster R-CNN R50 on COCO, as reported in the DySample paper (ICCV 2023), AP_S by upsampler:
  - Nearest neighbour: 21.3
  - CARAFE: 23.3
  - FADE: 23.1
  - SAPA-B: 22.4
  - DySample: 22.9
  - DySample+: 22.5
  - Overall AP is about 38.5 to 38.7 for the learned upsamplers. — [DySample arXiv 2308.15085](https://arxiv.org/pdf/2308.15085)
- DySample+ on R101 reaches 24.2 AP_S. — [DySample](https://arxiv.org/pdf/2308.15085)
- SAPA paper (Mask R-CNN table): CARAFE 23.0 AP_S, FADE 23.6 AP_S. — [SAPA arXiv 2307.08198](https://arxiv.org/pdf/2307.08198)
- CARAFE's original paper reports +1.8 AP_S over nearest-neighbour in Faster R-CNN, with fewer params/FLOPs than deconv or pixel shuffle. This is a secondary summary. — [CARAFE review](https://liner.com/review/carafe-contentaware-reassembly-features)
- LDA-AQU (2024) reports 39.2 AP, which is +0.5 over DySample and +0.6 over CARAFE in its own setup. AP_S was not in the snippet. — [LDA-AQU arXiv 2411.19585](https://arxiv.org/pdf/2411.19585)
- Repository for DySample: [github tiny-smart/dysample](https://github.com/tiny-smart/dysample)

**Foundation-model feature upsamplers (LoftUp, JAFAR; FeatUp, LiFT as baselines)**
- JAFAR (Couairon et al., 2025, arXiv 2506.11136) is an image-guided upsampler that works at any resolution. It is evaluated on semantic and open-vocabulary segmentation, depth, CAM and BEV segmentation. With DINOv2 ViT-S/14 it averages about +1.63 mIoU over FeatUp. No detection results. — [JAFAR arXiv](https://arxiv.org/pdf/2506.11136)
- LoftUp (Huang et al., ICCV 2025; github.com/andrehuang/loftup) is a coordinate-based cross-attention upsampler. It takes the high-resolution image, pixel coordinates and low-resolution VFM features, and is trained on pseudo-ground-truth from class-agnostic masks plus self-distillation. It is evaluated on segmentation, depth, normals, video object segmentation and interactive segmentation, against FeatUp and LiFT. No detection results. — [LoftUp arXiv](https://arxiv.org/html/2504.14032v1); [ICCV PDF](https://www.cvlibs.net/publications/Huang2025ICCV.pdf)

**DINOv3's own detection recipe (Plain-DETR, frozen backbone)**
- Setup:
  - Frozen DINOv3 7B/16 with a Plain-DETR adapter; the transformer encoder is a separate module.
  - About 100 M trainable params, all in the decoder.
  - Training: Objects365 at 1536 px for 22 epochs, then 1 epoch at 2048, then 12 COCO epochs at 2048. Inference at 2048.
- Results: 65.6 mAP, or 66.1 with multi-resolution TTA (1536 to 2880). No AP_S reported. — [DINOv3 arXiv HTML](https://arxiv.org/html/2508.10104)
- Community report (March 2026, Reddit): the Plain-DETR training code had not been released. An RF-DETR author said their DINOv2 ablations found frozen backbones "significantly worse" than unfrozen. Treat this as anecdotal. — [Reddit snapshot](https://reddit.sentinel-team.org/posts/1s3vtnu/snapshots/2026-03-28T05%3A27%3A13.534419Z)
- Distilled ViT-B, DINOv3 vs DINOv2 dense probes:
  - ADE20k: 51.8 vs 48.4 mIoU
  - NYU depth: 0.373 vs 0.416 (lower is better)
  - DAVIS: 77.2 vs 72.9
  - So DINOv3 ViT-B dense features are clearly better than DINOv2's. — [DINOv3 HTML, Table 14](https://arxiv.org/html/2508.10104)
- The DINOv3 teachers use 4 register tokens. Registers keep patch norms stable, and features "stay clean even at high resolutions". — [DINOv3 HTML](https://arxiv.org/html/2508.10104)

**DEIMv2 (2025, arXiv 2509.20787): real-time DETR on DINOv3 with a Spatial Tuning Adapter (STA)**
- Multi-scale features use no learnable parameters: the 1/16 features from blocks 5, 8 and 11 are bilinearly resized to several scales.
- STA adds an ultra-light conv branch for fine detail and fuses it with the ViT features via a 1x1-conv "Bi-Fusion". — [DEIMv2](https://arxiv.org/html/2509.20787v3)
- COCO results at 640 px:

  | Variant | Params | AP | AP75 | AP_S |
  |---|---|---|---|---|
  | X | 50.3 M | 57.8 | 63.2 | 39.2 |
  | L | 32.2 M | 56.0 | 60.9 | 37.5 |
  | S | 9.7 M | 50.9 | 55.1 | 31.4 |

- The backbone LR is much lower than the base LR (e.g. 2.5e-5 vs 5e-4 for S), which implies low-LR fine-tuning rather than freezing. No STA on/off ablation was found in the text. — [DEIMv2](https://arxiv.org/html/2509.20787v3)

**Patch size / resolution (RF-DETR, arXiv 2511.09554)**
- RF-DETR uses a DINOv2 backbone with patch size 16 and interleaved windowed and global attention. Patch size is interpolated FlexiViT-style during training.
- Smaller patches give higher accuracy at more compute. Higher resolution helps small objects at a runtime cost. — [RF-DETR HTML](https://arxiv.org/html/2511.09554v2)
- Swapping CAEv2 for DINOv2 gives about +2 AP. — [RF-DETR HTML](https://arxiv.org/html/2511.09554v2)

### Inferences
- Per parameter, a learned content-aware upsampler is the cheapest bet: about 50 to 200 k params for CARAFE or DySample on each up-step. It would replace the deconv / bilinear 2x and 4x steps in the SimpleFeaturePyramid (stride 16 to 8 to 4). CNN evidence suggests about +1.5 to 2 AP_S over nearest-neighbour. Whether the gain holds on top of transposed convolutions, which ViTDet already uses, is unverified.
- ViT-CoMer and DEIMv2-STA work by re-injecting CNN detail into, or fusing it with, ViT features. ViT-Adapter, a close relative, was negative here. That suggests the bottleneck is not missing a spatial prior per se, but possibly that frozen DINOv3 features plus DoRA cannot exploit the injected detail. Alternatively, the existing P2 stem already captures most of it.
- A cheaper STA-like variant may be worth testing within about 1 M params: take multi-layer taps (e.g. blocks 5/8/11), resize them bilinearly, and add a lightweight conv detail branch fused by 1x1 conv. Two caveats: block-level fusion was already negative, and DEIMv2 gives no isolated ablation.
- Patch-size interpolation, i.e. running the ViT at an effective patch size of 8 on the same 512 px image, is the only "input resolution" lever that does not enlarge the input. RF-DETR documents that smaller patches help accuracy. The cost is 4x tokens.
- LoftUp and JAFAR are plausible stride-4 feature sources for a frozen backbone. Without detection evidence they are speculative, and they add a separately trained module of several M params.

### Gaps
- No AP_S numbers were found for ViT-CoMer, ViT-Adapter or ViTDet.
- No detection (box AP) results were found for LoftUp, JAFAR, FeatUp or LiFT.
- DINOv3 Appendix D.9 (which layers feed Plain-DETR) could not be read. ViT-B Plain-DETR numbers do not appear to be published.
- No controlled study was found of patch-size interpolation, positional-embedding interpolation or register tokens on small-object AP with a frozen DINOv3.

## Q2. Feature pyramid / neck choices for small objects with plain ViT backbones

### Takeaway
With a plain ViT, ViTDet's simple pyramid is at least as good as FPN: +3.4 AP over no pyramid, against +2.5 to +3.1 for FPN variants. Putting effort into the stride-4 level is justified, since generating only a 1/4 map and subsampling it nearly matches the full simple pyramid. Cross-window propagation also matters (+1.7 to 2.3 AP), but it is a backbone-side change.

### Cited Findings
- ViTDet (ECCV 2022), Table 1: Mask R-CNN COCO AP_box (ViT-B / ViT-L):
  - No pyramid: 47.8 / 51.2
  - FPN 4-stage: 50.3 / 54.4
  - FPN last-map: 50.9 / 54.6
  - Simple pyramid (scales 1/32, 1/16, 1/8, 1/4): 51.2 / 54.6 — [ViTDet ar5iv](https://ar5iv.labs.arxiv.org/html/2203.16527)
- A variant that generates only the 1/4 map and subsamples it for the other levels reaches 54.5 AP on ViT-L, +3.3 over no pyramid. — [ViTDet ar5iv](https://ar5iv.labs.arxiv.org/html/2203.16527)
- Propagation (ViT-B), window-only baseline 48.9 AP:
  - 4 global blocks: 51.2
  - 4 conv blocks: 51.0
  - Shifted windows: 50.1
- Placement (ViT-L): evenly spaced 54.6 vs last-4 54.3 vs first-4 52.9. — [ViTDet ar5iv](https://ar5iv.labs.arxiv.org/html/2203.16527)
- DEIMv2 builds its pyramid with parameter-free bilinear resizing of multiple ViT blocks plus a light CNN detail branch, reaching 31.4 to 39.2 AP_S on COCO at 640 px. — [DEIMv2](https://arxiv.org/html/2509.20787v3)
- RF-DETR uses a multi-scale projector with LayerNorm instead of BatchNorm, chosen so that small-batch gradient accumulation works. The level ablation was not available. — [RF-DETR HTML](https://arxiv.org/html/2511.09554v2)
- EFPN (Deng et al.; IEEE TMM 2022 per search listings) adds an extra high-resolution level (P2'), built by a feature-texture-transfer (FTT) super-resolution module. It also uses cross-resolution distillation and a foreground-background-balanced loss, and claims SOTA on Tsinghua-Tencent 100K and COCO small. — [EFPN arXiv 2003.07021](https://arxiv.org/pdf/2003.07021)
  - Caution: the arXiv version was withdrawn by the authors in April 2020, who said "a part of important claims cannot stand solidly". — [arXiv abs](https://arxiv.org/abs/2003.07021)
  - A later version exists. — [ZJU PDF](https://april.zju.edu.cn/core/papercite-data/pdf/deng2022efp.pdf)

### Inferences
- With most objects under 32 px at 512 px input, P4 and P5 carry little useful signal. Moving neck capacity from P5/P6 to P2/P3 is consistent with ViTDet's finding that a 1/4-only generator is nearly as good. For example: reduce channels at P5 and drop the pool level, and give the P2 stem and P3 more channels or a content-aware upsampler.
- Switching from the simple pyramid to FPN, PAFPN or BiFPN on a plain ViT is unlikely to help much: ViTDet shows FPN adds nothing over the simple pyramid. Published plain-ViT evidence for BiFPN or PAFPN was not found.
- If the frozen backbone uses window attention, propagation blocks matter. DINOv3 ViT-B uses global attention, so this does not apply.

### Gaps
- No AP_S breakdown was found for ViTDet's pyramid ablation.
- No plain-ViT studies of PAFPN, BiFPN or attention necks with AP_S were found in this session.
- No numeric AP_S was retrieved for EFPN.

## Q3. Parameter-efficient fine-tuning for dense prediction with frozen DINOv2/v3

### Takeaway
The evidence is thin and mostly comes from segmentation, not COCO detection. LoRA on DINOv2 attention matches or beats full fine-tuning for BEV segmentation and gains up to rank 32. Detection-oriented recipes (RF-DETR, ViTDet, DEIMv2) fine-tune the whole backbone with layer-wise LR decay of 0.7 to 0.8 or a low backbone LR, rather than freezing it. One anecdotal report says frozen DINOv2 is significantly worse than unfrozen for detection.

### Cited Findings
- BEV segmentation with DINOv2:
  - Frozen DINOv2: 34.3 vs a 42.3 baseline.
  - LoRA in attention adds about 1 M params and reaches 42.3 (ViT-B) or 43.4 (ViT-L), beating full fine-tuning by 0.8.
  - Gains rise consistently with rank up to r=32. — [arXiv 2409.10228](https://arxiv.org/pdf/2409.10228)
- RF-DETR (DINOv2 backbone): per-layer multiplicative LR decay of 0.8 "to preserve earlier DINOv2 layers", base LR 1e-4, batch 128, gradient clip 0.1, EMA, no warm-up, scheduler-free. — [RF-DETR HTML](https://arxiv.org/html/2511.09554v2)
- ViTDet: layer-wise LR decay of 0.7 / 0.8 / 0.9 for ViT-B / L / H (gain up to 0.3 AP). Drop path 0.1 / 0.4 / 0.5. 100 epochs for B and L. Relative position bias adds about 1 AP. — [ViTDet ar5iv](https://ar5iv.labs.arxiv.org/html/2203.16527)
- DEIMv2 sets backbone LR far below the base LR (X: 1e-6 vs 5e-4; S: 2.5e-5 vs 5e-4). — [DEIMv2](https://arxiv.org/html/2509.20787v3)
- ViCo3D (2026, LiDAR 3D detection with an auxiliary DINOv2): unfreezing the top layers helps up to 4 blocks, then drops "significantly" at 12 (limited data). — [ViCo3D arXiv 2607.12959](https://arxiv.org/pdf/2607.12959)
- A DINOv3 change-detection paper applies LoRA to attention and MLP in the last 4 blocks only. — [arXiv 2511.19035](https://arxiv.org/pdf/2511.19035)
- Instance-segmentation PEFT study (2026, SEEM / Mask DINO, not DINOv2): full head tuning beats adding adapters in most settings. Tuning only the decoder or only the embeddings "leads to a significant decrease in AP". — [arXiv 2606.01947](https://arxiv.org/pdf/2606.01947)
- Anecdotal: RF-DETR authors found frozen DINOv2 significantly worse than unfrozen on COCO. — [Reddit snapshot](https://reddit.sentinel-team.org/posts/1s3vtnu/snapshots/2026-03-28T05%3A27%3A13.534419Z)

### Inferences
- For a pseudo-RGB SAR domain far from natural images, the strongest lever within the budget may be reallocating trainable capacity. Example: drop or shrink the Convpass 256 adapter (already found insensitive) and instead fully unfreeze the last 1 to 2 ViT-B blocks (about 7 M params each). Add layer-wise LR decay of about 0.7 to 0.8 on the remaining DoRA layers. This follows the ViTDet and RF-DETR practice and the ViCo3D "few top blocks" finding. It is not directly evidenced for SAR.
- A DoRA rank above 32 is unlikely to help given the BEV saturation at r=32. Moving DoRA from attention-only to attention + MLP, or concentrating it in later blocks, are untested alternatives.

### Gaps
- No COCO box-AP comparison of LoRA, DoRA, Convpass and last-block unfreezing on a frozen DINOv2/v3 detector was found.
- No study was found on which DINOv3 layers matter for localisation precision (AP75).

## Q4. Input-side methods within a fixed 512 px single pass

### Takeaway
Published evidence says resolution is the biggest small-object lever (RF-DETR, DINOv3 at 2048 px), but the 512 px cap rules that out. The in-budget equivalents are a smaller effective patch size (FlexiViT-style patch interpolation, documented in RF-DETR) and feature-level super-resolution (EFPN FTT, learned upsamplers).

### Cited Findings
- RF-DETR: "higher resolution improves small-object detection" at a runtime cost. Smaller patch sizes raise accuracy and compute, and patch-size interpolation is used in training. Small-object AP rises with model and resolution: N 25.2, S 32.0, M 36.1, 2XL 43.2 AP_S. — [RF-DETR HTML](https://arxiv.org/html/2511.09554v2)
- Multi-scale training range in RF-DETR is 0.5 to 1.5x with per-batch resize, plus only horizontal flips and random crops. — [RF-DETR HTML](https://arxiv.org/html/2511.09554v2)
- The EFPN FTT module super-resolves features and extracts regional detail at once, for an extra high-resolution level. Caveat: the arXiv version was withdrawn, see Q2. — [EFPN](https://arxiv.org/pdf/2003.07021)

### Inferences
- Running DINOv3 ViT-B with interpolated patch embeddings at patch 8 on a 512 px image gives a 64x64 token grid (stride 8). This is the closest legal substitute for higher input resolution, and it adds zero parameters. Attention compute and memory rise by 16x and tokens by 4x. A frozen backbone may degrade under patch-embedding resampling, so DoRA tuning would need to absorb that shift.
- Upsampling the image to 1024 px internally would probably violate the spirit of "input <= 512 px". Check the rule.

### Gaps
- No published small-object numbers for patch-size reduction on a frozen DINOv2/v3 detector were found.
- No image-level detail-enhancement study for SAR pseudo-RGB with ViT detectors was found.

## Q5. Training recipes affecting localisation (schedules, LSJ, copy-paste, distillation, self-training)

### Takeaway
Localisation distillation (LD) has the most direct high-IoU evidence: about +1.6 to 2.0 AP and +1.8 AP75 at zero inference cost. Copy-paste with LSJ gives large gains, especially in low-data regimes (+10 AP at 10% data). Copy-Blend is a newer variant that avoids fully overwriting the pasted region. Long schedules (100 epochs in ViTDet) and LSJ in [0.1, 2.0] are standard for plain-ViT detectors.

### Cited Findings
- **LD (Zheng et al., CVPR 2022; github.com/HikariTJU/LD):** distils the teacher's box-edge distributions, in GFocal's general distribution form, into the student over a "valuable localisation region".
  - GFocal R50 1x: 40.1 to 42.1 AP, with no inference cost. — [arXiv 2102.12252](https://arxiv.org/abs/2102.12252v4)
  - GFocalV2: +1.6 AP (41.1 to 42.7) and +1.8 AP75 (44.9 to 46.7), per a secondary summary. — [Liner review](https://liner.com/review/localization-distillation-for-dense-object-detection)
  - The CVPR 2022 supplement has further AP75 numbers. — [CVPR supp](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Zheng_Localization_Distillation_for_CVPR_2022_supplemental.pdf)
- **Simple Copy-Paste (Ghiasi et al., CVPR 2021):** pastes random objects with scale jitter. Fully occluded objects are removed and partially occluded boxes are trimmed.
  - Copy-Paste + LSJ ([0.1, 2.0] vs standard [0.8, 1.25]): +10 AP at 10% of COCO and +5 AP at full data (Mask R-CNN EfficientNet-B7 FPN, 640 px).
  - SOTA 57.3 box AP. — [Copy-Paste arXiv 2012.07177](https://arxiv.org/pdf/2012.07177)
- **Copy-Blend (DEIMv2):** blends pasted objects into the image rather than fully overwriting the target region. It is used with probability 0.5 in most epochs, alongside Mosaic and MixUp (p=0.5), which are turned off late in training. — [DEIMv2](https://arxiv.org/html/2509.20787v3)
- **ViTDet recipe:** LSJ [0.1, 2.0] at 1024 px, 100 epochs for B and L, layer-wise LR decay, drop path. — [ViTDet ar5iv](https://ar5iv.labs.arxiv.org/html/2203.16527)
- **DINOv3 detection:** large-scale pretraining of the detector on Objects365 before COCO. — [DINOv3 HTML](https://arxiv.org/html/2508.10104)
- **RF-DETR ablation:** Objects365 pretraining adds +0.7 AP (53.6 to 54.3). — [RF-DETR HTML](https://arxiv.org/html/2511.09554v2)

### Inferences
- The repo history ("copy-paste appears to cost") matches the label-corruption concern. Copy-Blend, or copy-paste that trims or occludes boxes properly, is the variant with published support. With a median of 2 objects per image, pasting into empty SAR background (sea or land clutter) with blending is the natural form.
- LD is the most targeted method for AP75 at a fixed parameter budget. The torchvision Faster R-CNN box head regresses deltas, not distributions, so LD would require either a distributional (GFL-style) box head or a smoothed-L1 / IoU-feature-imitation alternative. A larger teacher (e.g. DINOv3 ViT-L, trained without the parameter cap) is allowed because only the student ships.
- Detector pretraining on a larger SAR detection corpus (an analogue of Objects365 pretraining) is a documented gain source if external data is permitted.

### Gaps
- No LD results with ViT backbones or two-stage Faster R-CNN were retrieved.
- No AP_S-specific numbers for copy-paste or LSJ were found.
- Self-training / pseudo-labelling evidence for small objects was not searched, due to the tool budget.
- An isolated Copy-Blend vs Copy-Paste ablation was not found in the DEIMv2 text.
