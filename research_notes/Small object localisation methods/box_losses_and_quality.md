# Box regression losses, box representations and localisation-quality estimation for high-IoU AP in two-stage detectors (small-object emphasis)

Context: SAR imagery, frozen DINOv3 ViT-B/16 + DoRA/Convpass, ViTDet SFP + P2 stem, torchvision Faster R-CNN, smooth L1 + GIoU (GIoU w=1.0, box loss w=2.0), 512 px, 9 classes, median 2 objects/img, most < 32 px. Val: mAP .463 / AP50 .728 / AP75 .490 / AP_S .407. Negative so far: Cascade R-CNN, NWD, RPN IoU 0.6, RoI canonical scale 56, etc. Pending: IoU-aware ranking (score x IoU^0.5), GIoU w=2.0, box loss w=4.0, RoI sampling 256/0.5.

All numbers below are COCO val/minival or test-dev with ResNet-FPN backbones unless stated. None of these papers reports SAR or ViT results, so the gains transfer only roughly.

## 1. Box loss families (Smooth L1/L1 vs IoU family, Balanced L1, KL loss, Gaussian)

### Takeaway
On Faster R-CNN, swapping Smooth L1 for an IoU-family loss gives about +0.8 to +1.0 AP and +1.0 to +1.6 AP75. Among the IoU variants, the differences are small (≤0.7 AP). CIoU gives the best AP75 but lowers AP_S. DIoU or alpha-IoU (alpha=3) are the safest choices for small objects. Raising the regression loss weight past about 1.5x hurts AP_S in the published ablations, so the pending "box loss weight 4.0" ablation is risky for small objects. KL loss plus variance voting is the best-documented way to raise AP80/AP90 in a two-stage head at almost no cost. On R50-FPN, though, its AP75 gain is small and AP_S drops slightly.

### Cited Findings
**GIoU (Rezatofighi et al., CVPR 2019)**
- Faster R-CNN, COCO val: Smooth L1 .360 AP / .390 AP75; IoU .368 / .396; GIoU .369 / .398. Test-dev: Smooth L1 .364 / .392 → GIoU .373 / .404. Mask R-CNN val: .366/.397 → .376/.405. On VOC, Faster R-CNN AP75 rose .358 → .404 — [GIoU paper](https://ar5iv.labs.arxiv.org/html/1902.09630)
- GIoU *replaced* Smooth L1 only in the final RoI-head regression; the paper does not combine it with L1. The IoU/GIoU losses were multiplied by 10. The weight was tuned on VOC and "may be suboptimal on COCO" — [GIoU paper](https://ar5iv.labs.arxiv.org/html/1902.09630)
- The authors attribute the smaller COCO gain to dense anchors: Faster R-CNN has few non-overlapping cases, which is where GIoU's extra term helps — [GIoU paper](https://ar5iv.labs.arxiv.org/html/1902.09630)

**DIoU / CIoU / DIoU-NMS (Zheng et al., AAAI 2020)**
- Faster R-CNN R50-FPN, COCO val, regression loss weight 12 for all variants:

  | Loss | AP | AP75 | AP_S |
  |---|---|---|---|
  | IoU | 37.93 | 40.79 | 21.58 |
  | GIoU | 38.02 | 41.11 | 21.45 |
  | DIoU | 38.09 | 41.11 | 21.66 |
  | CIoU | 38.65 | 41.96 | 21.32 |
  | CIoU + DIoU-NMS | 38.71 | 42.07 | 21.37 |

  — [DIoU paper](https://ar5iv.labs.arxiv.org/html/1911.08287)
- CIoU lowers AP_S. The authors argue that for small objects the centre distance matters more than aspect ratio, and that the aspect-ratio term dilutes the distance term. DIoU-NMS does not fully recover the loss — [DIoU paper](https://ar5iv.labs.arxiv.org/html/1911.08287)

**Alpha-IoU (He et al., NeurIPS 2021; the venue is not stated on the fetched page)**
- Faster R-CNN R50-FPN, COCO val, 1x, MMDetection, alpha=3:

  | Loss | AP | AP75 | AP_S |
  |---|---|---|---|
  | IoU | 37.88 | 41.23 | 21.61 |
  | alpha-IoU | 38.96 | 41.94 | 22.14 |
  | GIoU | 37.62 | 41.00 | 21.49 |
  | alpha-GIoU | 39.18 | 42.00 | 22.05 |
  | alpha-DIoU | 39.43 | 42.18 | 22.10 |
  | alpha-CIoU | 39.25 | 41.92 | 21.94 |

  The mAP75:95 band rose 22.43 → 23.46 for GIoU → alpha-GIoU. The DIoU and CIoU baselines are copied from the original papers, so those comparisons are not like-for-like — [Alpha-IoU paper](https://ar5iv.labs.arxiv.org/html/2110.13675)

**Focal-EIoU (Zhang et al., Neurocomputing 2022)**
- Faster R-CNN R50-FPN, COCO val: original loss 37.3 AP / 40.3 AP75 / 21.3 AP_S → Focal-EIoU 38.9 / 42.4 / 21.2, with gamma=0.5. No per-variant Faster R-CNN ablation is reported.
- On RetinaNet, AP_S drops for IoU, GIoU, CIoU and EIoU (20.6–20.9 vs 21.2). Focal-EIoU gets back to 21.1 — [EIoU paper](https://ar5iv.labs.arxiv.org/html/2101.08158)

**Balanced L1 and loss-weight scaling (Libra R-CNN, Pang et al., CVPR 2019)**
- Faster R-CNN R50-FPN, COCO val. Baseline 35.9 AP / 38.4 AP75 / 21.2 AP_S.
  - Raising the Smooth L1 weight: 1.5x gives 36.4 / 39.7 / 20.8; 2.0x gives 36.2 / 39.5 / 20.2.
  - Plain L1: weight 1.0 gives 36.4 / 39.1 / 21.0; weight 2.0 gives 36.4 / 39.6 / 20.1.
  - Balanced L1 (alpha=0.5, gamma=1.5): 37.2 / 40.0 / 21.3.
  - Smaller alpha (0.2–0.3, gamma=1.0) gives the best AP_S (21.4–21.6) but less total AP — [Libra R-CNN](https://ar5iv.labs.arxiv.org/html/1904.02701)
- The authors attribute the drop at higher weights to outlier gradients — [Libra R-CNN](https://ar5iv.labs.arxiv.org/html/1904.02701)

**KL loss + variance voting (He et al., CVPR 2019)**
- The head adds one FC layer that predicts alpha = log(sigma^2) per coordinate. Its init has std 1e-4, so early training behaves like Smooth L1. It adds about 2 ms of latency — [KL-loss paper](https://ar5iv.labs.arxiv.org/html/1809.08545)
- Mask R-CNN R50-FPN, COCO:
  - Baseline: AP 38.6, AP80 34.4, AP90 11.3.
  - +KL: 39.5 / 36.0 / 15.8.
  - +KL + var voting: 39.9 / 36.4 / 17.0.
  - +KL + var voting + soft-NMS: 40.4 / 37.5 / 17.5.
  - AP50 falls 59.8 → 58.7 — [KL-loss paper](https://ar5iv.labs.arxiv.org/html/1809.08545)
- Faster R-CNN R50-FPN 2x:
  - Baseline: 37.9 / 41.1 AP75 / 21.5 AP_S.
  - +KL: 38.5 / 41.2 / 20.9.
  - +KL + vote: 38.8 / 41.6 / 21.0.
  - +KL + vote + soft-NMS: 39.2 / 42.5 / 21.2.
  - AP50 falls 59.2 → 57.6 — [KL-loss paper](https://ar5iv.labs.arxiv.org/html/1809.08545)
- VGG16 Faster R-CNN: AP75 22.8 → 30.4 and AP_S 6.7 → 8.7 with all three components. The gain is much larger on the weaker detector — [KL-loss paper](https://ar5iv.labs.arxiv.org/html/1809.08545)
- Voting uses sigma_t=0.02, with a usable range of 0.005–0.05. AP75/AP80/AP90 are sensitive to sigma_t — [KL-loss paper](https://ar5iv.labs.arxiv.org/html/1809.08545)

**Gaussian / NWD and small-object losses**
- NWD on Faster R-CNN, AI-TOD: 11.1 → 17.6 AP. This was negative in the user's own setup — [NWD arXiv 2110.13389](https://arxiv.org/pdf/2110.13389)
- Scale-Adaptive Loss (arXiv 2511.09891, 2025) reweights IoU losses to down-weight large objects. It reports +5.5 AP on YOLOv5/YOLOX only, with no Faster R-CNN result — [SAL](https://arxiv.org/html/2511.09891v1)
- ThermalIoU (IEEE SPL 2026): a heat-diffusion IoU loss evaluated on AI-TOD and VisDrone against CIoU, Inner-IoU and MPDIoU. No two-stage numbers were verified — [ThermalIoU](https://scholar.xjtu.edu.cn/en/publications/thermaliou-physical-theory-empowered-iou-loss-for-small-object-de/)

### Inferences
- Moving from GIoU to alpha-GIoU or alpha-DIoU (alpha=3) is a one-line change in torchvision's `fastrcnn_loss`: raise IoU to the power alpha before 1 - x and keep the penalty term. On Faster R-CNN it gave +1.2 to +1.6 AP and about +1 AP75 with a slightly higher AP_S. It is the cheapest loss change with clean two-stage evidence.
- Avoid CIoU's aspect-ratio term for tiny objects: AP_S dropped in both DIoU-paper and EIoU-paper ablations.
- Loss weights: published two-stage setups used IoU-family weights of about 10–12 when replacing Smooth L1. They did not combine IoU with L1, and their encoded deltas sit on a different scale from torchvision's (which uses weights 10,10,5,5). The user's L1 + GIoU combination with total weight 2.0 therefore has no direct published counterpart. Libra's results suggest that scaling the *L1* term above about 1.5x lowers AP_S, while scaling the IoU term is less clearly harmful. Prefer raising the GIoU weight (pending ablation) over raising the overall box weight to 4.0.
- KL loss adds about 9x4 outputs and almost no parameters. In torchvision, the box predictor emits an extra 4*num_classes log-variance and the loss becomes exp(-alpha)*smoothL1 + alpha/2. Variance voting is a post-NMS step that averages overlapping boxes weighted by exp(-(1-IoU)^2/sigma_t)/sigma^2.
  - Expect gains mainly at AP80/AP90. AP75 gains are small (+0.1 to +0.5 without soft-NMS on R50-FPN), and AP_S may drop about 0.5.
  - With a strong backbone the gain shrinks, as R50-FPN vs VGG16 shows.

### Gaps
- No peer-reviewed two-stage (Faster R-CNN) numbers were found for SIoU (arXiv 2022), Wise-IoU (arXiv 2023), Focaler-IoU (arXiv 2024), Inner-IoU (arXiv 2023) or MPDIoU (arXiv 2023). As far as could be checked, their evidence is mostly from YOLO-family experiments and blog reproductions. This session did not verify their exact tables, so they should be treated as unverified.
- No paper found ablates the L1:IoU weighting ratio for a *combined* L1+GIoU loss in a two-stage RoI head. That ratio comes from DETR-style practice (L1 5 : GIoU 2), which was not verified here for two-stage detectors.

## 2. Distribution and alternative box representations dropped into an RoI head

### Takeaway
Several RoI-head swaps have verified gains of about +2 to +3.3 AP and +2 to +4 AP75 over a Faster R-CNN FPN baseline, with modest AP_S gains of +0.8 to +1.5:
- SABL: best AP75 gain, about +10% runtime.
- Double-Head: conv head for regression.
- Dynamic R-CNN: zero parameters.
- Grid R-CNN: heavy.
- Libra R-CNN.

GFL/DFL has no verified two-stage numbers.

### Cited Findings
**SABL (Wang et al., ECCV 2020)**
- Faster R-CNN R50-FPN val: 36.4 → 39.7 AP; AP75 39.3 → 42.8; AP90 8.3 → 18.8; AP_S 23.1 with SABL. Params go 40.8M → 42M and FLOPs 212G → 270G for the default design — [SABL](https://ar5iv.labs.arxiv.org/html/1912.04260)
- Faster R-CNN R101 val: 38.5 / 41.6 AP75 / 22.3 AP_S → 41.6 / 45.0 / 23.5. AP50 drops 60.3 → 59.5. FPS goes 13.8 → 12.4 — [SABL](https://ar5iv.labs.arxiv.org/html/1912.04260)
- Ablation (R50, AP): box regression 36.4; boundary regression 37.3; bucketing only 32.8; bucketing + fine regression 38.3. Bucket-guided rescoring adds about 0.7 AP. Region scale factor sigma=1.7 is best (39.7) — [SABL](https://ar5iv.labs.arxiv.org/html/1912.04260)
- Cascade + SABL: 42.4 → 43.3 AP on test-dev — [SABL](https://ar5iv.labs.arxiv.org/html/1912.04260)

**Grid R-CNN (Lu et al., CVPR 2019)**
- R50-FPN minival: 37.4 / 40.3 AP75 / 21.8 AP_S → 39.6 / 42.4 / 22.6. The paper reports +4.1 AP at IoU 0.8 and +10.0 AP at IoU 0.9. AP50 drops 59.3 → 58.3 — [Grid R-CNN](https://ar5iv.labs.arxiv.org/html/1811.12030)
- Cost: 56x56 heatmaps per grid point, from 14x14 RoIAlign through 8 dilated convs and 2 deconvs; 125 RoIs at test time. Round objects gain less — [Grid R-CNN](https://ar5iv.labs.arxiv.org/html/1811.12030)
- Enlarging proposals hurt AP_S (22.1 → 20.8). Only the "extended region mapping" variant kept AP_S unchanged — [Grid R-CNN](https://ar5iv.labs.arxiv.org/html/1811.12030)

**Dynamic R-CNN (Zhang et al., ECCV 2020)**
- R50-FPN val:
  - Baseline: 37.0 AP, AP80 32.6, AP90 9.7.
  - +DLA: 38.0 / 34.4 / 13.2.
  - +DSL: 38.2 / 35.2 / 12.6.
  - Both: 38.9 / 36.3 / 15.2. AP50 falls 58.0 → 57.3.
  — [Dynamic R-CNN](https://ar5iv.labs.arxiv.org/html/2004.06002)
- Test-dev: 37.3 / 40.6 AP75 / 20.3 AP_S → 39.1 / 42.8 / 21.3 — [Dynamic R-CNN](https://ar5iv.labs.arxiv.org/html/2004.06002)
- How it adapts (C=100 iterations per update):
  - IoU threshold: every C iterations, set to the mean of the K_I-th largest proposal IoUs per image (K_I=75).
  - SmoothL1 beta: set to the median of the K_beta-th smallest regression errors (K_beta=10).
  - Smaller K_I favours high-IoU metrics — [Dynamic R-CNN](https://ar5iv.labs.arxiv.org/html/2004.06002)
- The text and table of the paper disagree on which component contributes which gain — [Dynamic R-CNN](https://ar5iv.labs.arxiv.org/html/2004.06002)

**Libra R-CNN (Pang et al., CVPR 2019)**
- R50-FPN val: baseline 35.9 / 38.4 AP75 / 21.2 AP_S. Each step adds to the previous:
  - +IoU-balanced sampling: 36.8 / 40.0 / 21.1.
  - +Balanced FPN: 37.7 / 40.9 / 22.4.
  - +Balanced L1: 38.5 / 42.0 / 22.9.
  — [Libra R-CNN](https://ar5iv.labs.arxiv.org/html/1904.02701)

**Double-Head R-CNN (Wu et al., CVPR 2020)**
- R50-FPN val: 36.8 / 40.4 AP75 / 21.2 AP_S → Double-Head 39.8 / 43.6 / 22.7 → Double-Head-Ext 40.3 / 44.2 / 22.4. R101: 39.1 / 42.4 / 22.2 → 41.5 / 45.6 / 23.8 — [Double-Head](https://ar5iv.labs.arxiv.org/html/1904.06493)
- The conv regression head (3 residual blocks + 2 non-local) has 7.19M params; the fc-head has 13.25M. Loss weights are w_fc=2.0 and w_conv=2.5 — [Double-Head](https://ar5iv.labs.arxiv.org/html/1904.06493)

### Inferences
- **Highest expected value for AP75 within budget: Double-Head and SABL.**
  - Double-Head: about +7M params. In torchvision, replace `TwoMLPHead` + `FastRCNNPredictor` with a split head: the fc branch handles cls, a conv branch on the 7x7 RoI feature handles reg.
  - SABL: about +1.2M params. Needs a custom box coder, since bucket targets replace deltas, and a custom loss (CE over buckets + Smooth L1 offsets). The mmdet implementation can be ported.
  - Both report AP_S gains of +1.2 to +1.6 on COCO.
- **Cheapest: Dynamic R-CNN.** It has no params and changes only `Matcher` thresholds and the SmoothL1 beta in `RoIHeads`. With a median of 2 objects per image, raising the IoU threshold may leave very few positive RoIs per image. Use a smaller-batch statistic, or a floor on the positive count. This connects to the negative RPN IoU 0.6 and Cascade results here, which also raise IoU bars. Note the user's GIoU term already removes most of the effect of beta.
- **Grid R-CNN is the weakest fit.** Its 56x56 heatmaps per RoI are costly, and a 512 px image with objects under 32 px gives very small RoIs.

### Gaps
- Generalized Focal Loss (DFL/QFL, NeurIPS 2020) and GFLV2 (CVPR 2021) were evaluated on one-stage detectors. No published two-stage RoI-head DFL ablation with AP75 numbers was found in this session. Localization Distillation (CVPR 2022) uses the distribution representation, but its two-stage numbers were not verified.
- TSD (CVPR 2020) and PISA (CVPR 2020) are relevant RoI-head changes, but their tables were not fetched this session.

## 3. Localisation-quality estimation and ranking

### Takeaway
The strongest verified two-stage gain over GIoU comes from ranking-based training: RS Loss gives 39.6 vs 37.6 AP and 43.0 vs 41.0 AP75 on R50-FPN, with sampling removed. IoU-Net's IoU-guided NMS raises AP80/AP90 by +2.7/+3.5 but costs AP50 (-2.0). That AP50 trade-off is a warning for the pending score x IoU^0.5 ablation.

### Cited Findings
**IoU-Net (Jiang et al., ECCV 2018)**
- FPN R50 minival: AP 36.4 → 37.3 with IoU-guided NMS; AP80 31.2 → 33.9; AP90 9.8 → 13.3; AP50 58.0 → 56.0. Soft-NMS alone gives only 36.8 — [IoU-Net](https://ar5iv.labs.arxiv.org/html/1807.11590)
- Optimisation-based refinement: FPN 36.4 → 38.0, AP90 9.8 → 14.6. The full pipeline on R101 gives 38.5 → 40.6. Runtime rises about 20% (0.255 → 0.305 s) — [IoU-Net](https://ar5iv.labs.arxiv.org/html/1807.11590)
- Refinement needs gradient ascent through PrRoI Pooling at test time, which means multiple backward passes — [IoU-Net](https://ar5iv.labs.arxiv.org/html/1807.11590)

**RS Loss (Oksuz et al., ICCV 2021)**
- Faster R-CNN R50-FPN 1x minival:
  - CE + Smooth L1: 36.5 / 39.4 AP75.
  - aLRP: 37.4 / 39.2.
  - GIoU: 37.6 / 41.0.
  - RS-R-CNN: 39.6 / 43.0.
  — [RS Loss](https://ar5iv.labs.arxiv.org/html/2107.11669)
- What RS-R-CNN changes:
  - Sigmoid classifiers trained with RS loss replace softmax.
  - All samplers are removed: the RPN uses all anchors and the R-CNN uses the top 1000 proposals. Removing the samplers alone gave about +1 AP.
  - GIoU is weighted by classification score, and the task weight is auto-balanced.
  - delta_RS=0.5; RPN weight 0.2; LR 0.012 — [RS Loss](https://ar5iv.labs.arxiv.org/html/2107.11669)
- RS-Mask R-CNN beats Mask Scoring R-CNN with no auxiliary head. Cascade R-CNN goes 40.3 → 41.3 — [RS Loss](https://ar5iv.labs.arxiv.org/html/2107.11669)
- aLRP converges slowly on two-stage detectors (14.4 vs 22.0 AP after 1 epoch) — [RS Loss](https://ar5iv.labs.arxiv.org/html/2107.11669)

**Variance as a quality signal (KL loss)**
- Variance voting uses the predicted uncertainty to merge boxes. See section 1 for the AP80/AP90 gains — [KL-loss paper](https://ar5iv.labs.arxiv.org/html/1809.08545)

### Inferences
- **What may beat score x IoU^0.5:**
  - RS loss: it trains the classifier to rank by IoU instead of fusing at test time. It is the best-documented two-stage option, at +2 AP and +2 AP75 over GIoU.
  - KL-loss variance plus voting: it uses localisation confidence to *move* boxes, not just re-rank them.
- **Implementation cost of RS loss in torchvision:**
  - Write a custom RS loss over sigmoid logits.
  - Set the RoI sampler to "all proposals" (huge `batch_size_per_image`, `positive_fraction` effectively ignored).
  - Use sigmoid classification.
  - This conflicts with the pending RoI sampling 256/0.5 ablation.
- With a median of 2 GT per image, sampling-free training puts very few positives against 1000 negatives. RS loss is designed to handle that imbalance, but SAR-specific evidence is missing.
- IoU-guided NMS and score x IoU fusion consistently cut AP50 by about 1–2 points while raising AP80/90. Watch AP50 in the pending ablation.

### Gaps
- VarifocalNet (CVPR 2021), Quality Focal Loss and centerness were reported on one-stage detectors. No verified two-stage AP75 numbers were found.
- No published numbers were found comparing score x IoU^p with p=0.5 against alternatives in a Faster R-CNN.

## 4. Iterative refinement without a full cascade

### Takeaway
Re-applying the same RoI regressor helps very little and degrades boxes above about 0.85 IoU. Cascade RPN, which improves proposals rather than the RoI stage, gives large AP75 and AP_S gains, but only after raising the R-CNN IoU threshold to 0.65 and cutting proposals to 300.

### Cited Findings
- **Iterative BBox (same regressor applied repeatedly), FPN+ COCO val:**
  - Baseline: AP 34.9, AP80 29.7, AP90 7.1.
  - Iterative BBox: 35.4 / 30.4 / 8.1.
  - Cascade R-CNN: 38.9 / 35.8 / 15.8.
  - "No benefit beyond applying f twice". A regressor trained at IoU 0.5 degrades boxes above about 0.85 IoU, and the error accumulates after three iterations.
  — [Cascade R-CNN paper](https://ar5iv.labs.arxiv.org/html/1712.00726)
- **Integral loss (shared regressor, several classifiers):** 35.4 AP; AP90 6.9 vs baseline 7.1 — [Cascade R-CNN paper](https://ar5iv.labs.arxiv.org/html/1712.00726)
- **Cascade RPN (Vu et al., NeurIPS 2019; the venue is not stated on the fetched page):**
  - Faster R-CNN R50-FPN test-dev: 37.1 / 40.1 AP75 / 21.4 AP_S with RPN → 40.5 / 44.2 / 22.6 with Cascade RPN.
  - AR100 rises 44.6 → 61.1; AR_S (100) rises 29.5 → 42.1. Runtime goes 0.04 → 0.06 s.
  - Getting the full gain required an R-CNN IoU threshold of 0.65 and 300 proposals; without these the gain was "limited".
  — [Cascade RPN](https://ar5iv.labs.arxiv.org/html/1909.06720)
- **IoU-Net optimisation-based refinement:** +1.6 AP on FPN. It needs test-time gradient steps — [IoU-Net](https://ar5iv.labs.arxiv.org/html/1807.11590)

### Inferences
- Applying the same head twice at inference costs about one extra RoI-head pass. Expect ≤ +0.5 AP and possible AP90 harm. It is not worth prioritising.
- Cascade RPN is single-pass and has a large reported AP_S/AR_S gain, which fits the small-object problem. It needs adaptive (deformable-offset) convs in the RPN and a higher R-CNN IoU threshold. That threshold change resembles the user's negative results (Cascade, RPN IoU 0.6), so it carries risk.
- IoU-Net refinement likely violates the "single forward pass" constraint.

### Gaps
- No evidence was found for training with shared-weight multi-stage refinement in a two-stage RoI head beyond the Cascade R-CNN paper's Iterative BBox baseline.

## 5. Post-processing for localisation

### Takeaway
Soft-NMS is free and gives about +0.7–1.1 AP and up to +1.3 AP75. Variance voting needs a KL-loss head and adds high-IoU AP. Single-model WBF is *worse* than NMS. DIoU-NMS adds only about +0.1 AP75.

### Cited Findings
- Soft-NMS improved Faster R-CNN COCO mAP by 1.1% with no retraining, at the same complexity as NMS — [Soft-NMS, ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Bodla_Soft-NMS_--_Improving_ICCV_2017_paper.pdf)
- R50-FPN Faster R-CNN 2x: soft-NMS raises AP 37.9 → 38.6, AP75 41.1 → 42.4 and AP_S 21.5 → 21.9 — [KL-loss paper](https://ar5iv.labs.arxiv.org/html/1809.08545)
- On FPN R50, soft-NMS adds +0.4 AP and AP90 +0.5, vs IoU-NMS at +0.9 AP and AP90 +3.5 — [IoU-Net](https://ar5iv.labs.arxiv.org/html/1807.11590)
- DIoU-NMS on top of CIoU: AP 38.65 → 38.71 and AP75 41.96 → 42.07 — [DIoU paper](https://ar5iv.labs.arxiv.org/html/1911.08287)
- WBF on a single model (RetinaNet, Open Images) scored mAP@0.5 0.4532 vs 0.4906 for NMS. The authors say WBF "works better when used for the models' ensemble". It is also about 3x slower than NMS — [WBF paper](https://ar5iv.arxiv.org/html/1910.13302)
- Variance voting with sigma_t=0.02 adds about +0.3–0.4 AP and +0.4 AP75 on R50-FPN, and +1.2 AP90 on Mask R-CNN — [KL-loss paper](https://ar5iv.labs.arxiv.org/html/1809.08545)

### Inferences
- Soft-NMS (Gaussian, sigma about 0.5) in torchvision replaces `batched_nms` in `postprocess_detections`. It needs a custom implementation; torchvision has none.
  - With a median of 2 objects per image and few overlapping objects, the gain may be smaller than on COCO.
  - It relies on a high `detections_per_img` (COCO eval allows 100) and a low score threshold (about 0.001–0.05) to raise recall-driven AP. Check that the eval config keeps these.
- Box voting without KL loss (score-weighted averaging of boxes in the NMS cluster) is a cheap single-model alternative. No two-stage numbers for it were verified here.

### Gaps
- Matrix NMS (SOLOv2) is designed for masks; no box-AP75 evidence was found for two-stage detectors.
- No source was found that quantifies the effect of score threshold or max detections on COCO AP75 for Faster R-CNN.
