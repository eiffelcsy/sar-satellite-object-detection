# Localisation and small-object detection in SAR and optical remote-sensing object detection

Scope note: about 18 search and fetch calls. Several key PDFs (DenoDet V2, SARATR-X-v2 and RFLA PDF builds) could not be text-extracted, so their HTML versions were used where available. No local PDF tooling (pdftotext or pypdf) was available. Everything in "Cited Findings" was read from the linked source. Recalled training knowledge appears only under Inferences or Gaps and is labelled.

Target context (from the brief): 9-class SAR, one class per image, a median of 2 objects per image, most objects under 32 px; frozen DINOv3 ViT-B/16 with DoRA and Convpass; pseudo-RGB of amplitude, BM3D and Sobel; ViTDet pyramid with a P2 stem; Faster R-CNN with GIoU at 512 px. Val scores: mAP .463, AP50 .728, AP75 .490, AP_S .407. Already negative: Cascade, NWD, mosaic, rot90 with jitter, ViT-Adapter and the WST stem.

## Q1. SAR benchmarks: SOTA numbers (mAP / AP50 / AP75 / AP_S) and which methods move AP75 and AP_S

### Takeaway
On SARDet-100K, mAP rises from about 49 (R50 Faster R-CNN) to about 57 (ConvNeXt-B MSFA, SARATR-X, DenoDet V2) and then to 63-69 (2026 large SAR foundation models). AP75 rises faster than AP50: SOTA is 93.9 AP50 and 75.3 AP75. Strong pretraining and better box-quality heads therefore show up mostly as high-IoU gains. Of the methods with both AP75 and AP_S reported and peer-reviewed or near-peer-reviewed, the largest gains come from three sources: better or SAR-specific backbones and pretraining, quality-aware dense heads (GFL family), and cascade refinement. Input-filter tricks such as MSFA move AP75 by only about 1 point.

### Cited Findings
**SARDet-100K (NeurIPS 2024 spotlight; 6 classes: Aircraft, Ship, Car, Bridge, Tank, Harbor; 116,598 images, 245,653 instances, 2.11 instances per image)**
- The dataset merges 10 SAR datasets, including AIR_SARShip, HRSID, MSAR, SADD, SAR-AIRcraft, ShipDataset, SSDD, OGSOD and SIVED. Instances per image range from 1.28 (ShipDataset) to 11.51 (SIVED). The mean Tank area is 461 px². — [SARDet-100K arXiv v3](https://arxiv.org/html/2403.06534v3)
- Detector sweep (R50, ImageNet-pretrained backbone, "INP" → MSFA), reported as mAP / AP50 / AP75 / AP_S — [SARDet-100K Table S8](https://arxiv.org/html/2403.06534v3):
  - Faster R-CNN: INP 49.0 / 82.2 / 52.9 / 43.5 → MSFA 51.1 / 83.9 / 54.7 / 45.2.
  - Cascade R-CNN: INP 51.1 / 81.9 / 55.8 / 44.9 → MSFA 53.9 / 83.4 / 59.8 / 47.2. Cascade gains +2.9 to +5.1 AP75 over Faster R-CNN but loses about 0.3-0.5 AP50.
  - Grid R-CNN: 48.8 / 79.1 / 52.9 / 42.4 → 51.5 / 81.7 / 56.3 / 45.1.
  - GFL: 49.8 / 80.9 / 53.3 / 42.3 → MSFA 53.7 / 84.2 / 57.8 / 47.8. GFL with MSFA has the best AP_S of the R50 detectors.
  - RetinaNet: MSFA 49.0 / 80.1 / 52.6 / 41.3.
  - FCOS: MSFA 48.5 / 82.1 / 51.4 / 42.9.
  - Deformable DETR: INP 50.0 / 85.1 / 51.7 / 44.0 → 51.3 / 85.3 / 54.0 / 44.9. It has the highest AP50 but a weak AP75.
  - DETR: 31.8 → 47.2 (+15.4) with MSFA.
- Backbone sweep (Faster R-CNN with MSFA), reported as mAP / AP50 / AP75 / AP_S — [SARDet-100K Table S9](https://arxiv.org/html/2403.06534v3):
  - R50: 51.1 / 83.9 / 54.7 / 45.2.
  - R101: 52.0 / 84.6 / 56.6 / 46.6.
  - ConvNeXt-T: 54.8 / 87.1 / 59.8 / 48.8.
  - ConvNeXt-B: 56.4 / 88.2 / 61.5 / 51.1.
  - Swin-B: 55.7 / 87.8 / 61.4 / 50.5.
  - VAN-B: 55.1 / 87.7 / 60.2 / 48.8.
  - Swin-T: 50.2 / 84.1 / 53.9 / 44.1.
  - Within the same detector, a stronger backbone gives about +7 AP75 and +6 AP_S.
- On SSDD and HRSID (Faster R-CNN with VAN-B), MSFA raises AP50 from 92.9 to 97.9 on SSDD and from 81.8 to 83.7 on HRSID. — [SARDet-100K](https://arxiv.org/html/2403.06534v3)
- SARATR-X (HiViT-B SAR foundation model, IEEE TIP 2025 per later citations; arXiv 2405.09365) reports 57.3 mAP / 88.7 AP50 / 62.8 AP75 on SARDet-100K, versus MSFA ConvNeXt-B at 56.4 / 88.2 / 61.5. — [SARATR-X arXiv](https://arxiv.org/pdf/2405.09365v5)
- SARMAE (ViT-B) reports 57.9 mAP on SARDet-100K. — [SARMAE arXiv 2512.16635](https://arxiv.org/html/2512.16635v1)
- SOTA table from SARATR-X-v2 (arXiv 2607.23238, 2026, GFL detector for HBB), mAP / AP50 / AP75 — [SARATR-X-v2](https://arxiv.org/html/2607.23238):
  - SARDet-100K:
    - SUMMIT ViT-B: 57.0 / 89.9 / 62.7.
    - SAMBA (Mamba): 59.7 / 89.1 / 68.0.
    - ViTP ViT-L: 59.7.
    - SARMAE ViT-L: 63.1.
    - BabelRS ViT-L: 63.3 / 91.7.
    - SARATR-X-v2 iTPN-B: 63.4 / 92.5 / 70.1.
    - SARATR-X-v2 iTPN-L: 68.7 / 93.9 / 75.3, which is +7.3 AP75 against only +2.2 AP50 over the previous best.
  - SSDD:
    - FEPS-Net R50: 59.9 / 96.0 / 67.5.
    - SARATR-X: 67.5 / 97.3 / 83.1.
    - SUMMIT: 70.4 / 96.7 / 86.3.
    - SAMBA: 71.3 / 97.8 / 85.6.
    - SARATR-X-v2-L: 71.8 / 97.0 / 87.6.
  - HRSID:
    - MLDet: AP50 92.8, AP75 53.4.
    - LFer-Net: AP50 90.6, AP75 78.7.
    - CV-SAR-FM Swin-B: 70.3 / 93.9 / 82.3.
    - SARATR-X-v2-B: 71.5 / 93.1 / 82.7.
    - SARATR-X-v2-L: 73.5 / 94.0 / 84.6.
  - The authors state that "the advantage of SARATR-X-v2 becomes more pronounced as the localization criterion tightens." No AP_S is reported.
- An alphaxiv summary gives SARATR-X results of 52.0 mAP on OGSOD and 58.7 on SAR-AIRcraft. — [alphaxiv summary](https://www.alphaxiv.org/abs/2405.09365). This is a secondary source.
- DenoDet V2 (arXiv 2508.09392, Aug 2025, GFL-based), SARDet-100K Table S7, reported as params / mAP / AP50 / AP75 / AP_S / AP_M / AP_L — [DenoDet V2](https://arxiv.org/html/2508.09392):
  - GFL: 32.27M / 55.01 / 85.16 / 58.87 / 49.44 / 67.29 / 60.45.
  - DenoDet V1: 65.78M / 55.88 / 85.81 / 60.16 / 50.63.
  - DenoDet V2: 32.60M / 56.71 / 85.86 / 61.55 / 51.45 / 68.75 / 61.34.
  - Cascade R-CNN scores 53.55 mAP, ATSS 54.95 and TOOD 54.65 in the same table.
  - Inconsistency: the same table lists Faster R-CNN at 39.22 mAP, which conflicts with the 49.0 in the SARDet-100K paper. Treat its Faster R-CNN row as an unreliable baseline.
- Other DenoDet V2 results: AIR-SARShip-1.0 AP rises from 72.42 (V1) to 73.98 (V2), and SAR-AIRcraft-1.0 mAP from 68.60 to 69.93. — [DenoDet V2](https://arxiv.org/html/2508.09392)
- In the original HRSID paper, SOTA detectors at the time scored over 84.7 AP50 and over 67.2 AP75. Inshore AP ranged from 41.3 to 61.8, and the authors note that adjacent ships challenge NMS. — [HRSID (IEEE Access 2020)](https://www.researchgate.net/publication/342541002_HRSID_A_High-Resolution_SAR_Images_Dataset_for_Ship_Detection_and_Instance_Segmentation)
- MSAR numbers are not comparable across sources:
  - DenoDet V1 reports 69.90 mAP (VOC07) and 71.21 (VOC12). — [DenoDet arXiv 2406.02833](https://arxiv.org/pdf/2406.02833)
  - A survey reports 97.2 mAP50 for DAFDet. — [SAR ATR survey arXiv 2509.22159](https://arxiv.org/pdf/2509.22159)
- FAIR-CSAR: single-look complex Gaofen-3 imagery at 1-5 m, 349,002 OBB instances. — [PASTE arXiv 2603.14886](https://arxiv.org/pdf/2603.14886); [FAIR-CSAR instructions](https://radars.ac.cn/fileLDXB/cms/news/info/2025/03/e1e05c53d9de439fb8f963360e78c0d4/Instruction%20of%20FAIR-CSAR-V1.0%20Dataset.pdf)
- SIVED: 1,044 chips and 12,013 rotated-box vehicles. — [SIVED, Remote Sensing 15(11):2825](https://api.crossref.org/works/10.3390%2FRS15112825). This is an MDPI journal.

### Inferences
- **AP75/AP50 ratio as a localisation-quality index:**
  - Target model: 0.490 / 0.728 = 0.67.
  - R50 Faster R-CNN with MSFA on SARDet-100K: 0.65.
  - ConvNeXt-B MSFA: 0.70.
  - DenoDet V2 (GFL): 0.72.
  - SARATR-X-v2-L: 0.80.
  - The target model's localisation quality per detected object is therefore close to a plain R50 Faster R-CNN. Box refinement is a real bottleneck, not just recall.
  - AP_S/mAP is 0.88 for the target model versus about 0.88-0.91 for SARDet-100K entries, so small objects are not disproportionately worse than in SARDet-100K.
- **Cascade:** on SARDet-100K it gives +2.9 to +5.1 AP75 with a trainable CNN backbone. The negative result in the target setting is probably specific to the setup: frozen backbone, smaller data, or one class per image with only about 2 GTs, which gives sparse positives at stage-3 IoU 0.7. It does not mean cascades never help on SAR.
- **GFL:** GFL-style quality-aware heads (Distribution Focal Loss and joint IoU-quality scoring) consistently beat Faster R-CNN on AP75 and AP_S on SARDet-100K: with MSFA, 57.8 vs 54.7 AP75 and 47.8 vs 45.2 AP_S. Inside a two-stage detector, the analogue is an IoU-prediction branch for rescoring (IoU-Net style) or distribution-based box regression in the RoI head. That is cheap (under 1M params) and transfers to frozen-ViT Faster R-CNN. This recommendation is inference; no SAR paper tested it in Faster R-CNN.

### Gaps
- SARATR-X-v2, SARMAE and SUMMIT report no AP_S on SARDet-100K.
- No AP75 or AP_S was found for SADD, MSAR, OGSOD, SIVED or FAIR-CSAR.
- No Papers-with-Code-style leaderboard was found that consolidates AP75 and AP_S. The PwC site has been sunset; hyper.ai mirrors exist but were not verified.
- The SARATR-X-v2 venue is unknown (arXiv, 2026).

## Q2. SAR-specific techniques that help localisation

### Takeaway
1. Handcrafted-filter pseudo-channels (MSFA) give about +1 AP75 and AP_S. The combination of HOG, Haar and WST beats any single filter.
2. Frequency-domain denoising in the backbone (DenoDet V2) gives about +2.7 AP75 and +2.0 AP_S for about 0.3M params, the best cost-to-gain SAR-specific module found.
3. Detection-level pretraining on optical remote sensing (DOTA) before SAR fine-tuning helps about +1 mAP.
4. Scattering-keypoint and edge methods exist for aircraft and inshore ships. The available evidence is mostly on MDPI or Chinese journals and rarely reports AP75.

### Cited Findings
- **MSFA input-filter ablation** (Faster R-CNN R50, mAP / AP50 / AP75 / AP_S) — [SARDet-100K Table 3/S7](https://arxiv.org/html/2403.06534v3):
  - SAR as RGB: 50.2 / 83.0 / 54.8 / 44.8.
  - With Canny: 50.7 / 83.6 / 55.0 / 45.3.
  - With HOG: 50.7 / 83.5 / 55.2 / 45.1.
  - With Haar: 50.6 / 83.4 / 54.7 / 45.4.
  - With WST: 51.1 / 83.9 / 54.7 / 45.2.
  - With GRE: 50.6 / 83.8 / 54.7 / 44.8.
  - With HOG + Haar + WST: 51.1 / 84.0 / 55.9 / 45.7, the best AP75 and AP_S.
  - The authors argue that filters close the domain gap. The pixel-space Pearson correlation between ImageNet and SAR statistics is 0.394, versus 0.99 for the HOG, Canny, Haar and WST feature spaces.
- **Pretraining domain (Faster R-CNN R50)** — [SARDet-100K Table 5](https://arxiv.org/html/2403.06534v3):
  - ImageNet backbone only: 49.0 mAP.
  - Multi-stage ImageNet then DOTA, whole framework: 50.2.
  - With WST, ImageNet only: 49.2. With WST and ImageNet then DOTA, framework: 51.1.
  - Pretraining the whole framework (neck and head), not just the backbone, on DOTA gives about +0.9 to 1.5 more than backbone-only.
  - DIOR helps less than DOTA (49.5 vs 50.2). The paper relates this to object size: mean instance area is 5,021 px² for DOTA and 12,726 px² for DIOR.
- **DenoDet V2 mechanism:** a DFT-domain plug-in in the backbone. It splits features into amplitude and phase, applies band-wise partitioned attention (stride 8) and Phase-Amplitude Token Exchange, then soft-thresholds before an inverse DFT. Cost is +0.33M params and +0.11 GFLOPs over GFL. — [DenoDet V2](https://arxiv.org/html/2508.09392)
- **DenoDet V2 ablations (SARDet-100K mAP)** — [DenoDet V2](https://arxiv.org/html/2508.09392):
  - Baseline: 55.0.
  - Amplitude-only refinement: 55.6. Phase-only: 56.2. Both: 56.4.
  - Adding token exchange: 56.7.
  - A stride-1 band (no partition) collapses to 52.5, so the band design matters.
- **SAR-AIRcraft-1.0** (Journal of Radars 2023, Gaofen-3 at 1 m spotlight): the dataset paper's method uses scattering keypoints to locate targets and a scatter-aware module to refine regression boxes for better localisation. — [SAR-AIRcraft-1.0](https://radars.ac.cn/en/article/doi/10.12000/JR23043)
- A physics-guided detector for SAR airplanes is on arXiv; no numbers were extracted. — [arXiv 2411.12301](https://arxiv.org/pdf/2411.12301)
- A 2026 arXiv paper on scattering structure reports that a topology loss L_topo gives +2.5 AP and +3.6 AP75. This is on KP-SAR-Aircraft-1.0, a keypoint benchmark, not on SAR-AIRcraft-1.0 detection. — [arXiv 2606.06847](https://arxiv.org/html/2606.06847)
- Related work: Scattering-Enhanced Attention Pyramid Network for SAR aircraft (TGRS 2021). — [ADS abstract](https://ui.adsabs.harvard.edu/abs/2021ITGRS..59.7570G/abstract). SFEF-Net (scattering feature fusion; Sensors, MDPI). — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12114894/)
- **Edge enhancement for ships:** a morphological-network edge-enhancement method helped less on HRSID. The authors say HRSID "contains many small ships" that are hurt by morphological operations. — [Sensors/PMC 11243990](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11243990/) (MDPI)
- **Other edge and dense-ship papers** — [AMANet arXiv 2401.13214](https://arxiv.org/pdf/2401.13214); [SMEP-DETR, Remote Sensing 2025](https://doi.org/10.3390/rs17060953); [closely-arranged inshore ships, IJRS 2023](https://www.tandfonline.com/doi/full/10.1080/01431161.2023.2277166):
  - AMANet: HRSID 68.9 AP and 91.4 AP50.
  - SMEP-DETR (multi-edge enhancement): HRSID 93.2 mAP50.
  - Bi-directional attention FPN for closely arranged inshore ships: SSDD 0.902 and HRSID 0.839 mAP.
  - None reports AP75.
- **Multitask Gaussian-mask joint segmentation for SAR ships:** dense supervision via Gaussian masks. The PDF was too large to fetch, so its numbers were not verified. — [arXiv 2411.13847](https://arxiv.org/pdf/2411.13847)

### Inferences
- **Filter channels:** the target model's pseudo-RGB already uses amplitude, BM3D and Sobel. MSFA evidence says Sobel or Canny-type edges add the least AP75 (+0.2). HOG and the HOG + Haar + WST combination add the most (+1.1 AP75, +0.9 AP_S). Two swaps are worth testing:
  - Replace Sobel with a HOG magnitude map.
  - Replace BM3D with a Haar-wavelet band.
  - Expected effect is about 1 point at most. The WST stem was negative for the target model, consistent with WST alone not helping AP75 (54.7 vs 54.8).
- **DenoDet V2:** its frequency-denoising block is the most transferable SAR-specific module. It is a plug-in on feature maps, so it can sit on the ViTDet pyramid levels (P2-P5) or in the adapter path without touching the frozen ViT. Its cost is about 0.3M params per placement, well within the 40M trainable budget. Its gain was measured with a trainable CNN backbone in GFL, so transfer to frozen ViT features is unproven.
- **Detection-level pretraining:** pretraining the full detector (neck, RPN, RoI head and adapters) on DOTA or DIOR HBB, then fine-tuning on SAR, is a cheap transferable idea for a frozen-backbone model. The adapters and heads are exactly what gets pretrained. DOTA's smaller objects made it the better source.

### Gaps
- No peer-reviewed paper reports AP75 for shadow or layover handling, or for speckle-statistics losses (e.g. Gamma or Rayleigh priors) in detection.
- Strong-scatterer or CFAR-prior proposal methods with COCO-style AP75 were not found within budget.
- AP75 evidence for edge or contour supervision in two-stage detectors on HRSID or SSDD was not found. Most papers report only AP50 or mAP.

## Q3. Optical remote-sensing small-object methods

### Takeaway
On AI-TOD, Gaussian or receptive-field label assignment (RFLA and successors) gives large gains on very tiny objects (under 8-16 px) for two-stage detectors. Most of that gain is recall at low IoU. The target model already found NWD negative, and its objects are mostly 16-32 px, so the expected payoff is limited.

### Cited Findings
- RFLA (ECCV 2022) replaces IoU-based assignment with a Gaussian receptive-field distance plus hierarchical assignment, and reports +4.0 AP over SOTA on AI-TOD. — [RFLA ECCV](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690518.pdf)
- A blog (non-peer-reviewed) quotes these AI-TOD gains at 12 epochs: Faster R-CNN 11.1 → 21.1 and DetectoRS 14.8 → 24.8. — [savant-ai blog](https://savant-ai.io/blog/small-objects-detection/)
- A third-party summary says Faster R-CNN AP_vt rises from 0.0 to 9.5. — [Lacuna summary](https://lacuna.tiptreesystems.com/work/rfla-gaussian-receptive-field-based-label-assignment-for-tiny-object-detection/wrk_9bb714c202df4bf1dc36d2f3157fe99e)
- Similarity-Distance label assignment reports +10.5, +5.8 and +1.8 AP over DotD, NWD and RFLA on AI-TOD. — [arXiv 2407.02394](https://arxiv.org/html/2407.02394v1)
- Normalized-Gaussian label assignment reports 25.7 AP on AI-TOD-v2 and 27.9 on VisDrone. — [Remote Sensing 18(3):396 (MDPI)](https://www.mdpi.com/2072-4292/18/3/396)

### Inferences
- Label-assignment metrics (NWD, RFLA, DotD) target objects under about 16 px, where IoU between anchors and GTs is near zero. The target model's AP_S is 0.407 with most objects under 32 px, and NWD was already negative. That suggests most objects are in the 16-32 px band, where IoU assignment works and localisation precision, not assignment, is the limit.
- **Unverified training knowledge**, from memory rather than this session's sources; verify before citing. Methods that improved AP75 or AP_S on DOTA/DIOR/AI-TOD and fit Faster R-CNN:
  - CFINet (ICCV 2023): coarse-to-fine RPN plus feature imitation, on SODA-D and SODA-A.
  - Dynamic Anchor Learning and Oriented R-CNN for oriented boxes.
  - HBB vs OBB: OBB helps dense, rotated ships and vehicles.

### Gaps
- No verified numeric AP75 or AP_S tables were extracted for DIOR, xView, NWPU VHR-10 or LEVIR in this session. The RFLA PDF fetch failed, so AP75 and AP_vt per method were not verified.
- The effect of oriented versus horizontal boxes on HBB AP75 was not found with numbers.

## Q4. Pretraining and foundation models: do they help high-IoU localisation versus DINOv2/v3?

### Takeaway
SAR-specific foundation models now dominate SARDet-100K. SARATR-X-v2 iTPN-L reaches 75.3 AP75, and its gains concentrate at high IoU. No source in this session compares them against frozen DINOv2 or DINOv3 for SAR detection. Optical remote-sensing foundation models (SkySense, RVSA/MTP) beat DINOv2 on optical remote-sensing tasks, including DIOR, per their own papers.

### Cited Findings
- SARATR-X-v2 (iTPN-L) improves on the previous SOTA by +2.2 AP50 but +7.3 AP75 on SARDet-100K, and by +1.3 AP75 on SSDD (with −0.8 AP50). The gains are attributed to better localisation. — [SARATR-X-v2](https://arxiv.org/html/2607.23238)
- At ViT-B or HiViT-B scale, SAR foundation models cluster at 57-58 mAP and 62.6-62.8 AP75 on SARDet-100K: SARATR-X, SUMMIT and SARMAE. That is only about +1 over ConvNeXt-B with ImageNet and MSFA (56.4 / 61.5). — [SARATR-X-v2 table](https://arxiv.org/html/2607.23238); [SARMAE](https://arxiv.org/html/2512.16635v1)
- SARATR-X-v2 iTPN-B reaches 63.4 mAP and 70.1 AP75. — [SARATR-X-v2](https://arxiv.org/html/2607.23238)
- SkySense (CVPR 2024) reports superiority over DINOv2 on five Earth-observation datasets, including DIOR, and attributes DINOv2's gap to domain shift. — [SkySense arXiv 2312.10115](https://arxiv.org/pdf/2312.10115)
- SkySense V2 (ICCV 2025, 665M params) reports 64.41 on DIOR-R, 75.24 on DOTA and 96.22 on HRSC2016. — [SkySense V2](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_SkySense_V2_A_Unified_Foundation_Model_for_Multi-modal_Remote_Sensing_ICCV_2025_paper.pdf)
- RVSA-L (MTP) beats a ViT-G foundation model on DOTA-v2 and uses the ViTDet pyramid. — [MTP arXiv 2403.13430](https://arxiv.org/html/2403.13430v1); [RVSA arXiv 2208.03987](https://arxiv.org/pdf/2208.03987)

### Inferences
- The SAR foundation-model jump from about 57 to 63 mAP happens at "B" scale only in 2026 models (SARATR-X-v2 iTPN-B with about 70 AP75). Those numbers come from full fine-tuning, not a frozen backbone. For a frozen-backbone, 130M-total-parameter setup, SARATR-X(-v2) B-size weights are a candidate swap for DINOv3 ViT-B. HiViT-B and iTPN-B are about 90M params, within budget. They should be tested frozen with the same adapters. Expect the largest effect on AP75, if any.
- Risk: these models were pretrained on SAR data that may overlap the target benchmark's sources (SARDet-100K components). Check for leakage.

### Gaps
- No published comparison of frozen DINOv2 or DINOv3 against SAR foundation models on SAR detection AP75 or AP_S was found. AFRL-DINOv2 is named in SARATR-X-v2 but its detection numbers were not visible.
- The SatMAE and Scale-MAE detection AP75 numbers were not retrieved. From recalled knowledge, Scale-MAE reports mainly classification and segmentation, not detection.

## Q5. Pitfalls: augmentations and methods that hurt SAR detection

### Takeaway
Evidence is thin and mostly from MDPI or low-tier venues. Copy-paste in SAR is usually done box-wise (BoxPaste), and its benefit varies. Strong compositing augmentations (copy-paste, CutMix, mosaic) can shift the training distribution away from real SAR test data. Mosaic helped on a saturated SSDD AP50 metric, which says little about AP75 or small objects.

### Cited Findings
- BoxPaste (Remote Sensing 2022, MDPI) pastes box-cropped ships rather than mask-cropped ones. Its argument: SAR sea backgrounds are homogeneous enough. — [BoxPaste](https://www.mdpi.com/2072-4292/14/22/5761)
- Ship-Go (ISPRS J. 2023) argues that copy-paste, CutMix and mosaic shift the training distribution away from real data, and proposes diffusion inpainting instead. — [Ship-Go](https://www.sciencedirect.com/science/article/abs/pii/S0924271623003350)
- An adversarial-training SAR ship paper (MDPI Sensors) found copy-paste scored lower AP50 than mosaic on average across detectors, with detector-dependent reversals. — [PMC 11243904](https://pmc.ncbi.nlm.nih.gov/articles/PMC11243904)
- H2Det reports mosaic raising SSDD AP from 96.5 to 99.1. — [H2Det](https://www.researchgate.net/publication/356679024_H2Det_A_High-Speed_and_High-Accurate_Ship_Detector_in_SAR_Images)
- Morphological edge operations hurt small ships on HRSID. — [PMC 11243990](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11243990/)
- DenoDet V2's frequency module collapses from 56.7 to 52.5 mAP if not band-partitioned. The design detail is critical. — [DenoDet V2](https://arxiv.org/html/2508.09392)
- SARDet-100K: on small SAR datasets (SSDD), larger backbones overfit, e.g. R152 below R101. — [SARDet-100K](https://arxiv.org/html/2403.06534v3)

### Inferences
- The target benchmark has one class per image and about 2 objects. That makes mosaic and copy-paste unrealistic in the same way Ship-Go describes: they create multi-object, multi-background scenes absent from the test set. This is consistent with the target model's negative mosaic result and the recent commit noting that copy-paste "appears to cost".
- Rotation: SAR look direction fixes shadow and layover orientation. Arbitrary rot90 creates illumination geometry inconsistent with the sensor. Horizontal flip along azimuth is the safest geometric augmentation. This is reasoning, not found in a cited paper.

### Gaps
- No controlled peer-reviewed study was found that measures AP75 or AP_S under mosaic, copy-paste or rotation for SAR detection. All evidence is AP50 on saturated ship datasets.
