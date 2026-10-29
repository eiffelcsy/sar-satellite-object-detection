# Label assignment, anchors and proposals, sampling and RoI feature extraction for small and tiny objects in two-stage detectors

Context for the reader: SAR imagery; frozen DINOv3 ViT-B/16 with DoRA and Convpass adapters; ViTDet simple pyramid with a P2 stem; torchvision Faster R-CNN with smooth L1 + GIoU; fused-max RoIAlign over P2-P5; 512 px input; median 2 objects per image; most objects are under 32 px. Current val: mAP .463, AP75 .490, AP_small .407. Remaining budget is about 1 M trainable parameters. Already negative: Cascade R-CNN, NWD, mosaic, EMA, ViT-Adapter, RoI canonical scale 56, RPN fg IoU 0.6.

Metric definitions: AI-TOD splits objects into very tiny (vt, 2-8 px), tiny (t, 8-16 px), small (s, 16-32 px) and medium (m, 32-64 px). SODA uses extremely small (eS), relatively small (rS), generally small (gS) and normal (N).

Coverage note: this pass used about 15 tool calls. The label assignment numbers come from the papers' own tables. Several RoI-extraction and sampling items had no small-object evidence I could reach; those are listed under Gaps rather than filled in from memory.

## Q1. Label assignment for tiny objects (RFLA, SimD, NWD-RKA, DotD, ATSS, Dynamic R-CNN, CFINet dynamic thresholds); why IoU starves small objects

### Takeaway
On tiny-object benchmarks, replacing IoU with a Gaussian or normalised-distance similarity in the anchor/RPN assigner, combined with top-k ranking, is the largest single gain reported for Faster R-CNN. On AI-TOD, RFLA goes from 11.1 to 21.1 AP and SimD to 23.9 AP. SimD also raises AP75 from 7.6 to 17.6 on AI-TOD and from 24.2 to 28.1 on SODA-D. Most of the gain is on objects under 16 px. On 16-32 px objects the gain is small (+2.8 to +4.7 AP_s), and AP_m can drop. Both methods add zero parameters.

### Cited Findings
- **RFLA (ECCV 2022)** models each feature point's receptive field as a Gaussian. It scores that Gaussian against the GT box with a Receptive Field Distance (KLD by default), assigns the top-k (k=3), then runs a second, relaxed stage for GT boxes left without positives (Hierarchical Label Assignment, stage factor β=0.9). It replaces both IoU thresholds and centre sampling. — [RFLA arXiv html](https://arxiv.org/html/2208.08738v2)
- RFLA on AI-TOD, Faster R-CNN R50-FPN, 12 epochs:

  | | AP | AP50 | AP75 | AP_vt | AP_t | AP_s | AP_m |
  |---|---|---|---|---|---|---|---|
  | Baseline | 11.1 | 26.3 | 7.6 | 0.0 | 7.2 | 23.3 | 33.6 |
  | w/ RFLA | 21.1 | 51.6 | 13.1 | 9.5 | 21.2 | 26.1 | 31.5 |

  — [RFLA](https://arxiv.org/html/2208.08738v2)
- RFLA with other detectors on AI-TOD — [RFLA](https://arxiv.org/html/2208.08738v2):
  - Cascade R-CNN: 13.8 → 22.1 AP.
  - DetectoRS: 14.8 → 24.8 AP. DetectoRS with NWD reaches 20.8.
  - RetinaNet: only 8.7 → 9.1 AP.
- RFLA on other datasets (Faster R-CNN) — [RFLA](https://arxiv.org/html/2208.08738v2):
  - VisDrone2019: 22.3 → 23.4 AP, 38.0 → 41.4 AP50, AP_vt 0.1 → 4.8, AP_t 6.2 → 11.7.
  - DOTA-v2.0: 35.6 → 36.3 AP.
  - TinyPerson: AP50_tiny 48.7 → 50.1.
  - The gain shrinks sharply when the dataset is not dominated by objects under 16 px.
- RFLA ablations on AI-TOD — [RFLA](https://arxiv.org/html/2208.08738v2):
  - RFD alone gives 20.7 AP; adding HLA gives 21.1 AP.
  - The distance choice inside HLA matters: GIoU 17.9 AP, Wasserstein 21.1, KLD 21.1.
  - Top-k from 1 to 4 gives 20.7, 21.1, 21.1 and 20.9 AP, so it is insensitive to k.
  - β=0.8 drops to 19.7 AP.
- RFLA's mechanism claim: for tiny GT boxes, almost every anchor has IoU = 0, and no anchor point falls inside the box. Such objects get few or no positive samples, and the paper attributes the +9.6 AP from RFD to fixing this. — [RFLA](https://arxiv.org/html/2208.08738v2)
- **SimD (IROS 2024)** has the form `exp(-(sim_location + sim_shape))`. Both terms are centre and size differences normalised by (w_g+w_a) and (h_g+h_a). The normalisers m and n are averages computed over the training set, so there are no hand-set hyperparameters. It is used in place of IoU in RPN and R-CNN assignment with thresholds pos 0.7, neg 0.3, min-pos 0.3. In the experiments, NMS stays IoU-based (0.7 for RPN, 0.5 for R-CNN). — [SimD ar5iv](https://ar5iv.arxiv.org/html/2407.02394); [arXiv abs](https://arxiv.org/abs/2407.02394v1)
- SimD results, Faster R-CNN R50-FPN, baseline → SimD — [SimD](https://ar5iv.arxiv.org/html/2407.02394):

  | Dataset | AP | AP50 | AP75 | AP_vt | AP_t | AP_s | AP_m |
  |---|---|---|---|---|---|---|---|
  | AI-TOD | 11.1 → 23.9 | 26.3 → 54.5 | 7.6 → 17.6 | 0 → 11.9 | 7.2 → 24.8 | 23.3 → 28.0 | 33.6 → 33.5 |
  | AI-TODv2 | 12.8 → 24.5 | | 9.4 → 17.2 | | | 24.6 → 28.9 | 37.0 → 35.4 |
  | VisDrone2019 | 22.3 → 26.5 | 38.0 → 47.9 | | 0.1 → 7.5 | 6.2 → 16.3 | | |
  | SODA-D | 28.9 → 31.1 | 59.7 → 59.8 | 24.2 → 28.1 | | | | |

  Blank cells were not reported.
- SimD normalisation ablation on AI-TOD: none 20.4 AP / 13.2 AP75; width only 22.1; height only 22.0; width and height 23.9 AP / 17.6 AP75. — [SimD](https://ar5iv.arxiv.org/html/2407.02394)
- SimD's claimed margins over prior methods are +10.5 AP over DotD, +5.8 over NWD and +1.8 over RFLA. Several of the comparison rows use a DetectoRS base rather than Faster R-CNN, so the comparison is not like-for-like. — [SimD abstract](https://arxiv.org/abs/2407.02394v1); [ar5iv tables](https://ar5iv.arxiv.org/html/2407.02394)
- **NWD-RKA (ISPRS J. 2022)** uses Normalized Wasserstein Distance with ranking-based (top-k) assignment instead of IoU thresholds. It reports +4.3 AP over state-of-the-art competitors on AI-TOD-v2 with DetectoRS. AI-TOD-v2 is a relabelled AI-TOD. I could not retrieve the Faster R-CNN per-module table. — [arXiv 2206.13996](https://arxiv.org/abs/2206.13996)
- **DotD (CVPRW 2021)** reaches 16.1 AP / 10.6 AP75 / 8.3 AP_vt on AI-TOD in RFLA's table, but only 18.1 AP_s against a 23.3 Faster R-CNN baseline. Its AP_s and AP_m fall below baseline. — [RFLA table](https://arxiv.org/html/2208.08738v2)
- **ATSS** (as a one-stage detector) gets 12.8 AP, 1.9 AP_vt and 19.5 AP_s on AI-TOD — [RFLA](https://arxiv.org/html/2208.08738v2). On SODA-D it gets 26.8 AP, below Faster R-CNN's 28.9 — [CFINet](https://arxiv.org/html/2308.09534).
- **Dynamic R-CNN (ECCV 2020)** raises the R-CNN positive IoU threshold during training based on proposal IoU statistics, and shrinks the SmoothL1 β based on regression-error statistics. The code is in MMDetection v2.2. Results on COCO, R50-FPN — [arXiv 2004.06002](https://arxiv.org/pdf/2004.06002); [ECVA](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2335_ECCV_2020_paper.php):

  | Setting | AP | AP90 |
  |---|---|---|
  | Baseline | 37.0 | 9.7 |
  | Dynamic SmoothL1 only | 38.0 | 13.2 |
  | Dynamic label assignment only | 38.2 | 12.6 |
  | Both | 38.9 | 15.2 |

  The combined gain is +1.9 AP and +5.5 AP90, with no inference cost.
- **CFINet CRPN dynamic anchor threshold (ICCV 2023).** An anchor is positive for coarse regression if IoU > T_a, where T_a = max(0.25, 0.20 + 0.15·log(√(wh)/12)). Smaller GT boxes therefore get a lower IoU bar. On SODA-D, fixed T_a of 0.20 and 0.40 give 29.9 and 30.1 AP, against 30.3 for the dynamic rule. — [CFINet](https://arxiv.org/html/2308.09534)
- Lowering the RPN positive IoU from 0.7 to 0.5 barely changes proposal recall on SODA-D (AR 41.2 → 41.3, AR_eS 24.0 → 24.2). — [CFINet Table 4](https://arxiv.org/html/2308.09534)
- An "Adaptive Dynamic Label Assignment" paper (IEEE JSTARS 2024) proposes per-object dynamic thresholds and evaluates on AI-TOD and LEVIR-Ship. I did not retrieve its numbers. — [bohrium listing](https://bohrium.dp.tech/paper/arxiv/948929991817560088)
- **DCFL (CVPR 2023, oriented, one-stage)** combines a learnable dynamic prior (DCN) with coarse positives from Generalized Jensen-Shannon Divergence across FPN levels, then posterior re-ranking and a dynamic GMM filter. It reports state of the art on DOTA-v1.5, DOTA-v2 and DIOR-R. I did not retrieve its AI-TOD numbers. — [CVPR poster](https://cvpr.thecvf.com/virtual/2023/poster/22569)

### Inferences
- **The baselines inflate the headline gains.** The AI-TOD baselines use mmdet default anchors: on P2, octave base scale 8 × stride 4 gives a 32 px anchor. With those anchors, a 6 px box has IoU near 0 with everything. This is why AP_vt starts at 0.0 and why +10-13 AP is possible.
- **Expect much less on this project's data.** Objects here sit mainly in the 8-32 px range, with AP_s already 0.407. The relevant reference numbers are therefore AP_t (+14-17) and AP_s (+2.8 to +4.7), not overall AP. Whether the current torchvision anchors already go down to about 8-16 px on P2 is the key unknown. If they do, RFLA or SimD will mainly change how many positives each GT gets, and the gain could be close to zero.
- **The NWD result is a warning but not decisive.** NWD was already tried here and was negative. RFLA and SimD are close relatives: RFLA's own ablation shows Wasserstein ≈ KLD. That is a warning sign for RFLA-KLD. Two things still differ from a plain NWD swap:
  - SimD normalises by object size, which is scale-invariant, unlike NWD's fixed constant C.
  - Top-k ranking with a fallback stage for unmatched GTs is a different mechanism from thresholding a new metric.
  - If the NWD trial only swapped the metric at fixed thresholds, a top-k/HLA-style assigner is still untested.
- **Dynamic R-CNN targets AP75 directly and adds zero parameters.** With a median of 2 objects per image, raising the R-CNN IoU threshold may starve the head of positives. That is the same failure that probably sank Cascade R-CNN here. A safer variant is Dynamic SmoothL1 alone (+1.0 AP, +3.5 AP90 on COCO); "SmoothL1 β" means the transition point of the smooth L1 loss. It could also be combined with GIoU.
- **Implementation in torchvision.** RFLA-style and SimD-style assignment means replacing `box_ops.box_iou` in `RegionProposalNetwork.assign_targets_to_anchors` with a custom similarity and a top-k matcher. This is about 50-100 lines, uses no parameters, and has negligible training cost. Replacing the IoU in R-CNN `assign_targets_to_proposals` is optional. SimD's m and n are two scalars computed once from the training annotations.

### Gaps
- No published Faster R-CNN numbers isolate RFLA or SimD in the RPN versus the R-CNN stage. SimD does not ablate where it is applied.
- I found no evidence for these assigners on SAR data, or on a frozen-ViT/ViTDet backbone. All results above use ImageNet-pretrained R50-FPN.
- NWD-RKA Faster R-CNN AP75 and AP_s on AI-TOD-v2: tables not retrieved.
- OTA/SimOTA used inside a two-stage RPN: I found no published two-stage tiny-object result.
- CenterAssign: not covered in this pass.
- I found no 2025-2026 label-assignment paper that clearly beats SimD on AI-TOD with Faster R-CNN. A targeted search returned only 2022-2024 work.

## Q2. RPN and proposals: anchor sizes, Cascade RPN, Guided Anchoring, coarse-to-fine RPN, proposal counts and NMS

### Takeaway
Improved RPNs give 1-3 AP on COCO. On small-object benchmarks the gain is small: Cascade RPN gives +0.2 AP on SODA-D, and CFINet's CRPN gives +1.4 AP. Proposal recall for extremely small objects stays near 24% AR at 300 proposals whichever RPN is used. Cascade RPN even lowers AR_eS. The best-evidenced cheap RPN change is a size-dependent (lower for small boxes) positive IoU threshold, as in CFINet's CRPN.

### Cited Findings
- **Cascade RPN (NeurIPS 2019)** uses two-stage anchor refinement with adaptive (alignment-guided dilated) convolution. It reports +13.4 AR over RPN. On COCO it takes Faster R-CNN from 37.1 to 40.6 AP and Fast R-CNN from 37.0 to 40.1. — [arXiv 1909.06720](https://arxiv.org/pdf/1909.06720); [NeurIPS](https://papers.nips.cc/paper/2019/hash/d554f7bb7be44a7267068a7df88ddd20-Abstract.html)
- In their NeurIPS rebuttal, the Cascade RPN authors concede that the gains are larger for large objects. They note GA-RPN shows the same pattern. — [author feedback](https://media.nips.cc/nipsbooks/nipspapers/paper_files/nips32/reviews/825-AuthorFeedback.pdf)
- Cascade RPN on SODA-D: 29.1 AP / 25.9 AP75 / 12.5 AP_eS, against Faster R-CNN's 28.9 / 24.1 / 13.8. AP75 improves, but extremely small objects get worse. — [CFINet](https://arxiv.org/html/2308.09534)
- **Guided Anchoring (CVPR 2019)** reports 9.1% higher recall with 90% fewer anchors, and +2.7% mAP for Faster R-CNN on COCO. In the MMDetection zoo, GA-Faster R-CNN R50-FPN reaches 39.6 AP and GA-RPN R50 reaches AR1000 68.4. — [mmdet GA README](https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/guided_anchoring/README.md)
- Proposal recall on SODA-D at 300 proposals per image — [CFINet Table 4](https://arxiv.org/html/2308.09534):

  | Proposal method | AR | AR_eS | AR_rS | AR_gS |
  |---|---|---|---|---|
  | RPN (pos IoU 0.7) | 41.2 | 24.0 | 38.3 | 47.3 |
  | RPN-0.5 | 41.3 | 24.2 | | |
  | GA-RPN | 42.1 | 24.1 | | |
  | Cascade RPN | 41.8 | 22.8 | | |
  | CRPN | 42.6 | 24.6 | 38.9 | 49.1 |

  Blank cells were not reported.
- **CFINet (ICCV 2023)** = CRPN (coarse regression over P2-P5 with the size-dependent IoU threshold, then an adaptive-conv RPN) + Feature Imitation (a contrastive auxiliary branch). Ablation on SODA-D — [CFINet](https://arxiv.org/html/2308.09534):

  | Setting | AP | AP_eS | AP_rS | AP_gS |
  |---|---|---|---|---|
  | Baseline | 28.9 | 13.8 | 25.7 | 34.5 |
  | + CRPN | 30.3 | 14.3 | 27.3 | 36.1 |
  | + FI | 29.5 | | | |
  | + both | 30.7 | 14.7 | 27.8 | 36.4 |

  Full CFINet AP75 is 26.7 against 24.1.
- On SODA-A (oriented), CFINet reaches 34.4 AP against 32.5 for Rotated Faster R-CNN. — [CFINet](https://arxiv.org/html/2308.09534)
- RFLA's AI-TOD setup uses 3000 RPN proposals, and test-time NMS 0.5 with up to 3000 boxes. SimD keeps RPN NMS at 0.7. — [RFLA](https://arxiv.org/html/2208.08738v2); [SimD](https://ar5iv.arxiv.org/html/2407.02394)
- RFLA's "Gaussian anchor" (19.6 AP) and "receptive anchor" (18.9 AP) variants show that replacing the anchor prior alone recovers most of the RFLA gain over 11.1 AP. RFLA argues this is more robust than tuning anchor sizes by hand. — [RFLA](https://arxiv.org/html/2208.08738v2)

### Inferences
- With a median of 2 objects per image, torchvision's defaults (2000/1000 proposals pre/post NMS for train/test) far exceed what is needed. Proposal count is unlikely to be the bottleneck. The recall ceiling comes from scoring and regression of tiny anchors.
- CRPN-style size-dependent IoU thresholds for RPN positives cost no parameters. In torchvision this means replacing the Matcher's fixed `fg_iou_thresh` with a per-GT threshold. The previous negative result came from *raising* the threshold to 0.6; CRPN evidence points the other way, lowering it for small GTs to about 0.25.
- Cascade RPN and GA-RPN add an adaptive or deformable conv and an extra head stage. Within a 1 M budget, a single 256-channel 3×3 conv is about 0.59 M parameters, so only one is affordable. The small-object evidence for both is weak (+0.2 AP and negative AR_eS on SODA-D).

### Gaps
- I found no published ablation of anchor sizes 4-16 px versus 32 px on P2 for Faster R-CNN tiny-object benchmarks. RFLA argues qualitatively that this is brittle.
- I found no study of pre/post-NMS proposal counts or RPN NMS thresholds for sparse (≤ 5 objects) images.
- Anchor-free RPNs (e.g. CenterNet2's first stage) on small-object benchmarks: not retrieved.

## Q3. RoI feature extraction (RoIAlign size, context RoIs, level mapping, PrRoI, deformable RoI)

### Takeaway
I found little direct small-object evidence on RoI extraction in this pass. One study found that enlarging the support region to 2× the RoI area gave minor or no gain for regular and aligned RoI pooling. A 2025 context-aware RoI pooling paper reports +1.3 AP_vt and +1.5 AP_t. These items should be treated as low-evidence.

### Cited Findings
- "Learning Region Features for Object Detection" (ECCV 2018) compares three support regions: the RoI itself, the RoI enlarged to 2× area, and the whole image. Larger support helps its learned region-feature module, but "has minor and no improvements for regular and aligned RoI pooling". — [arXiv 1803.07066](https://arxiv.org/pdf/1803.07066)
- A 2025 tiny-object paper with a context-aware RoI pooling layer (parallel local and global-context branches) reports AP50 55.2 → 57.0, AP_vt 9.3 → 10.6 and AP_t 24.8 → 26.3. Its baseline numbers match RFLA-DetectoRS on AI-TOD, so it likely builds on that. — [arXiv 2506.09897](https://arxiv.org/html/2506.09897v1)
- CFINet's Feature Imitation branch adds a supervised-contrastive auxiliary loss that pulls small-object RoI features toward high-quality (IoU > 0.65) exemplars. It gives +0.6 AP on SODA-D alone and +0.4 on top of CRPN, at train-time-only cost. — [CFINet](https://arxiv.org/html/2308.09534)

### Inferences
- In this project, about 95% of RoIs would map to P2 under the standard rule, and fused max over P2-P5 is already used. RoIAlign at 14×14 on P2 means a 16 px object spans about 4×4 P2 cells. Upsampling 4×4 cells to 7×7 or 14×14 mostly interpolates.
- Context enlargement (about 1.5× RoI, concatenated or added to the fc input) is cheap in torchvision: run a second MultiScaleRoIAlign on scaled boxes, then add a 1×1 projection. It is plausible for SAR, where wakes, shadows and surroundings are informative. It has weak published support for standard RoIAlign.

### Gaps
- No small-object (AP_s) ablation found for:
  - RoIAlign output 7 versus 14, or sampling ratio.
  - PrRoI Pooling.
  - Deformable RoI pooling.
  - PANet adaptive feature pooling on small objects.
- The AP_S column of "Learning Region Features" Table 2 was not retrieved.

## Q4. RoI head design (2fc versus conv, Double-Head, TSD) and sampling (OHEM, IoU-balanced, PISA, positive fraction)

### Takeaway
Head decoupling and sampling gains are documented only on COCO. The head designs give about +3 AP (Double-Head +3.5 AP, TSD about +3 mAP), and the sampling methods +2 AP (Libra +2.5 AP overall, PISA +2.0 AP). None has published small-object or sparse-image evidence that I could retrieve. Double-Head's conv regression branch targets localisation (AP75), but at full size it far exceeds a 1 M parameter budget.

### Cited Findings
- **Double-Head R-CNN (CVPR 2020)** finds that the fc-head is better for classification, being more spatially sensitive, and the conv-head is better for box regression. It uses an fc-head for classification and a conv-head for regression. Gains over FPN on COCO are +3.5 AP (R50) and +2.8 AP (R101); the MMDetection R50 1x model reaches 40.0 AP. — [mmdet double_heads README](https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/double_heads/README.md)
- **TSD (CVPR 2020)** derives separate task-specific proposals (spatial offsets) for classification and regression from a shared proposal. It reports about +3% mAP across backbones on COCO and OpenImage, plus about +1% from a progressive constraint. — [CVF open access](https://openaccess.thecvf.com/content_CVPR_2020/html/Song_Revisiting_the_Sibling_Head_in_Object_Detector_CVPR_2020_paper)
- **Libra R-CNN (CVPR 2019)** combines IoU-balanced sampling, a balanced feature pyramid and balanced L1 loss. It gives +2.5 AP over FPN Faster R-CNN on COCO, and Libra Faster R-CNN R50 1x reaches 38.3 AP. The README gives no per-component split. — [mmdet libra README](https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/libra_rcnn/README.md)
- **PISA (CVPR 2020)** focuses training on "prime samples", which are the high-IoU positives and highest-ranked negatives. It changes training only, with no inference cost. Faster R-CNN R50-FPN goes from 36.4 to 38.4 AP (+2.0), and the paper reports about +2% over random sampling, OHEM and Focal Loss. — [mmdet PISA README](https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/pisa/README.md)

### Inferences
- **PISA fits this project's constraints best.** It needs zero parameters, adds only a loss re-weighting at train time, and is explicitly aimed at high-IoU positives (AP75). Its re-weighting makes sense when positives are few, as with 2 GTs per image.
- **IoU-balanced sampling targets negatives.** It only matters if hard negatives are scarce under random sampling. In sparse SAR images with clutter, this could matter. Cost is about 30 lines in torchvision's `BalancedPositiveNegativeSampler`.
- **A light Double-Head variant fits the budget.** A single 256-channel 3×3 conv (about 0.59 M parameters) plus pooling as a separate regression branch, with the existing 2fc kept for classification, fits about 1 M. The full Double-Head conv head (several 1024-channel residual blocks) does not. This variant is untested.
- **The pending 256/0.5 RoI sampling change has a likely limit.** With a median of 2 GTs, the number of positives is capped by proposals with IoU ≥ 0.5 to those 2 GTs, which is typically tens. A 0.5 positive fraction will often not be filled, and the sampler then pads with negatives. A gain from that change would most likely come from batch size rather than the positive fraction.

### Gaps
- OHEM: no small-object numbers retrieved in this pass.
- No numbers retrieved for:
  - PISA, Libra or Double-Head on AP_S, AP75 or tiny-object datasets.
  - RoI batch size or positive fraction tuning for sparse images.
  - TSD's exact Faster R-CNN table.
- No 2025-2026 two-stage RoI-head paper specific to tiny objects was retrieved, except the context-RoI paper in Q3. A candidate is "MAFE R-CNN: Selecting More Samples to Learn Category-aware Features for Small Object Detection", which appeared in search; its review page returned HTTP 429, so its numbers are unverified.
