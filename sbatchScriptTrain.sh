#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ## 
#################################################

#SBATCH --nodes=1                   # How many nodes required? Usually 1
#SBATCH --cpus-per-task=4           # Number of CPU to request for the job
#SBATCH --mem=32GB                   # How much memory does your job require?
#SBATCH --gres=gpu:1                # Do you require GPUS? If not delete this line
#SBATCH --time=01-00:00:00          # How long to run the job for? Jobs exceed this time will be terminated
                                    # Format <DD-HH:MM:SS> eg. 5 days 05-00:00:00
                                    # Format <DD-HH:MM:SS> eg. 24 hours 1-00:00:00 or 24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL  # When should you receive an email?
#SBATCH --output=%u.%j.out          # Where should the log files go?
                                    # You must provide an absolute path eg /common/home/module/username/
                                    # If no paths are provided, the output file will be placed in your current working directory
#SBATCH --constraint=l40s

################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=student                 # The partition you've been assigned
#SBATCH --account=student   # The account you've been assigned (normally student)
#SBATCH --qos=studentqos       # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=eiffelchong.2023@scis.smu.edu.sg # Who should receive the email notifications
#SBATCH --job-name=EiffelJob     # Give the job a name

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Purge the environment, load the modules we require.
# Refer to https://violet.scis.dev/docs/Advanced%20settings/module for more information
# module purge
# module load Python/3.11.7

# Create a virtual environment can be commented off if you already have a virtual environment
# python3.11 -m venv ~/myenv

# This command assumes that you've already created the environment previously
# We're using an absolute path here. You may use a relative path, as long as SRUN is execute in the same working directory
source .venv/bin/activate

# If you require any packages, install it as usual before the srun job submission.
# pip3 install numpy

# Submit your job to the cluster.
DATA=/common/scratch/users/e/eiffelchong.2023/cs701-sar-course-data/train/

# One DINOv3 run is roughly 1.5-2 h; the 12 runs together do not fit one 24 h job, so split them over jobs.
# Base (control; rerun it so every comparison uses the same code):
# srun --gres=gpu:1 python train.py --config configs/dinov3_dora_giou.yaml --data "$DATA" --wandb

# A: one change each versus the base (a6 = Cascade on the warm-started a4; compare it with a4):
# srun --gres=gpu:1 python train.py --config configs/ablations/a1_input.yaml             --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/a2_aug.yaml               --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/a3_wst.yaml               --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/a4_warmstart.yaml         --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/a5_fusion.yaml            --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/a6_warmstart_cascade.yaml --data "$DATA" --wandb --workers 4

# B: capacity, each on top of a1 and filling the 40 M trainable budget a different way (compare with a1):
# srun --gres=gpu:1 python train.py --config configs/ablations/b1_full_blocks.yaml  --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/b2_rank_early.yaml   --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/b3_rank_uniform.yaml --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/b4_convpass.yaml     --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/b5_hybrid.yaml       --data "$DATA" --wandb --workers 4

# C: combinations of the winners (b4 Convpass; a4 warm start vs a6 warm start + Cascade) and their control.
# Same --workers 4 as the a*/b* runs. Read them as: c1a vs b4_convpass_58ep = the warm start's own effect (same 58
# epochs); c1b vs c1a = Cascade on top (check both the local held-out and Codabench val); c2 vs b3 / b4 = mixing
# DoRA rank and Convpass. Roughly 2-2.5 h each (58-epoch runs the longest), ~9-10 h for all four in sequence.
# srun --gres=gpu:1 python train.py --config configs/ablations/c1a_b4_warmstart.yaml         --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/c1b_b4_warmstart_cascade.yaml --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/b4_convpass_58ep.yaml         --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/c2_rank64_convpass192.yaml    --data "$DATA" --wandb --workers 4

# D: longer training of the best model (b4_convpass_58ep: Codabench 19.16), with and without the WST stem.
# Compare b4_convpass_72ep with b4_convpass_58ep (does a longer schedule keep paying?) and b4_wst_72ep with
# b4_convpass_72ep (does WST complement Convpass?). Roughly 3 h each, ~6 h for both in sequence.
# srun --gres=gpu:1 python train.py --config configs/ablations/b4_convpass_72ep.yaml --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/b4_wst_72ep.yaml      --data "$DATA" --wandb --workers 4

# E: one change each versus the current best, b4_convpass_72ep (Codabench 21.80). ~3 h per 72-epoch run (d1 ~4 h):
# the nine runs (~28 h) exceed one 24 h job, so submit batch 1, then comment it out and uncomment batch 2.
# Before (or alongside) these, run the post-processing sweep in sbatchScriptEval.sh on the existing 72-epoch model.
# Batch 1 (schedule + head reallocation, ~16 h):
# srun --gres=gpu:1 python train.py --config configs/ablations/d1_96ep.yaml           --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/d2_ema.yaml            --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/d3_head512_cp352.yaml  --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/d4_convhead_cp352.yaml --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/d5_depthwise640.yaml   --data "$DATA" --wandb --workers 4
# Batch 2 (Convpass variants + detector, ~12 h):
# srun --gres=gpu:1 python train.py --config configs/ablations/d6_multiscale.yaml     --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/d7_parallel.yaml       --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/d8_rpn06.yaml          --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/d9_stem128.yaml        --data "$DATA" --wandb --workers 4

# G: after the d* results (d1_96ep best: local mAP 0.396; d9's stem the only other positive). ~13 h in total.
#   e1 vs d1 = does d9's stem stack on 96 epochs; e2 vs d1 = does 120 epochs still pay; e3 vs d1 = seed noise
#   (same held-out images, data.split_seed 0), which says whether the e1 / e2 gaps are real.
# srun --gres=gpu:1 python train.py --config configs/ablations/e1_stem_96ep.yaml   --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/e2_120ep.yaml       --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/e3_d1_seed1.yaml    --data "$DATA" --wandb --workers 4

# H: stem + longer schedules (Codabench: d1 26.65, e1 stem 28.33, e2 120 epochs 29.95). ~11.5 h in total.
#   e4 vs e1 / e2 = do the stem and 120 epochs add up; e5 vs e4 = does 150 epochs still pay. Submit both to
#   Codabench: a local gain for e5 that does not carry over there is the sign of overfitting.
# # srun --gres=gpu:1 python train.py --config configs/ablations/e4_stem_120ep.yaml  --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/e5_stem_150ep.yaml  --data "$DATA" --wandb --workers 4

# I: the longest schedule so far, and the final model on all of train (Codabench: e4 120 epochs 30.51, e5 150 epochs
# 32.55). They are independent, so run them as two jobs at the same time: submit this script with only the e6 line
# active, then comment it out, uncomment the d10 line and submit again.
#   e6 vs e5 = does 200 epochs still pay (~8.7 h); d10 = the e5 recipe on all of train, the safe final submission
#   (~7.3 h).
# # srun --gres=gpu:1 python train.py --config configs/ablations/e6_stem_200ep.yaml  --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/d10_full_train.yaml --data "$DATA" --wandb --workers 4

# J: learning rate and weight decay, each one change on e1_stem_96ep (stem, 96 epochs; local mAP 0.400, Codabench
# 28.33), ~4 h each. Independent: submit this script once per line, with only that line active.
#   f1 adapters 3e-4 | f2 adapters 1e-3 + 2-epoch warmup | f3 heads and adapters 2e-4 | f4 no decay on 1-D params
# srun --gres=gpu:1 python train.py --config configs/ablations/f1_adapter_lr3e-4.yaml --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/f2_adapter_lr1e-3.yaml --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/f3_lr2e-4.yaml         --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/f4_wd_exclude_1d.yaml  --data "$DATA" --wandb --workers 4

# K: the split learning rates (adapters 3e-4, detection heads 2e-4, classifier 1e-4) at 96 and 200 epochs. Independent:
# submit this script once per line, with only that line active.
#   f5 (~4 h) vs f1 / f3 = do the detection gain and the F1 come together; g1 (~8.7 h) vs e6 = does it hold at 200
#   epochs.
# srun --gres=gpu:1 python train.py --config configs/ablations/g1_split_lr_200ep.yaml --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/f5_split_lr.yaml       --data "$DATA" --wandb --workers 4

# L: the final model. g1's recipe at 250 epochs on all of train (~12 h); no held-out split, so its Codabench score
# against d10 (38.50) is the only check. Make the zips with sbatchScriptEval.sh section 5.
# srun --gres=gpu:1 python train.py --config configs/ablations/d11_full_train_g1_250ep.yaml --data "$DATA" --wandb --workers 4

# M: the final model. g1's exact recipe (200 epochs) on all of train (~9.7 h); d11 at 250 epochs tied d10 (38.34 vs
# 38.50). No held-out split: compare its Codabench score with d10's. Make the zips with sbatchScriptEval.sh section 6.
# srun --gres=gpu:1 python train.py --config configs/ablations/d12_full_train_g1.yaml --data "$DATA" --wandb --workers 4

# N: small objects (held-out at 200 epochs: AP_small 0.38 vs AP_medium 0.58, AP_large 0.67; no change so far moved
# AP_small beyond seed noise). One change each on f5_split_lr (96 epochs, g1's rates; ~4 h), compared with f5:
#   h1 ViT-Adapter instead of the P2 stem | h2 8 px tokens (several times slower: time one epoch first, and check the
#   rule with the TA) | h3 max(IoU, NWD) label assignment. Independent: submit once per line, with only that line active.
# srun --gres=gpu:1 python train.py --config configs/ablations/h1_vit_adapter.yaml --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/h2_patch8.yaml       --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/h3_nwd.yaml          --data "$DATA" --wandb --workers 4
#   h4 IoU branch, detections ranked by class score x predicted IoU^0.5 (localization: AP50 0.73 vs AP75 0.47). ~4 h.
#   Afterwards sweep --iou-alpha 0 0.25 0.5 1 2 with sbatchScriptEval.sh section 8 (alpha 0 = its own control).
# srun --gres=gpu:1 python train.py --config configs/ablations/h4_iou_aware.yaml    --data "$DATA" --wandb --workers 4
#   h5 LoRA+ (DoRA B matrices at 8x the adapter rate; no new parameters) | h6 training-only P2 center-heatmap loss
#   (0.15 M). ~4 h each, compared with f5.
# srun --gres=gpu:1 python train.py --config configs/ablations/h5_lora_plus.yaml    --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/h6_p2_heatmap.yaml   --data "$DATA" --wandb --workers 4

# O: trade macro-F1 for mAP (delta-m: +1 mAP ~ +1.97, -1 F1 ~ -0.53). One change each on g1 (200 epochs, ~8.7 h),
# compared with g1: i1 classification loss weight 0.25 | i2 no classification gradient into the backbone.
# srun --gres=gpu:1 python train.py --config configs/ablations/i1_cls_loss025_200ep.yaml  --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/i2_cls_detach_200ep.yaml  --data "$DATA" --wandb --workers 4

# P: final candidates from O, 250 epochs on all of train (~12 h each; independent, so submit one job per line):
#   d13 = i2 (classifier stop-gradient) | d14 = i1 at classification weight 0.1. Make the zips with
#   sbatchScriptEval.sh section 9 and compare their Codabench delta-m with d12.
# srun --gres=gpu:1 python train.py --config configs/ablations/d13_full_train_i2_250ep.yaml       --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/d14_full_train_i1_cls01_250ep.yaml --data "$DATA" --wandb --workers 4

# Q: new base j0 (f5 + classification 0.1, no mosaic, same-class copy-paste, 1 px margin; 96 epochs, ~4 h) and one
# change each on it. Two at a time: round 1 = j0 + j1, round 2 = j2 + j3. Compare j1-j3 with j0.
#   j1 RoI pooled from all levels (max) | j2 RoI canonical scale 56 | j3 classifier on [CLS, mean token]
# srun --gres=gpu:1 python train.py --config configs/ablations/j0_base.yaml            --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/j1_roi_fused.yaml       --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/j2_roi_canonical56.yaml --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/j3_cls_token.yaml       --data "$DATA" --wandb --workers 4
# d15: d14 + j0's augmentation, 250 epochs on all of train (~12 h); vs d14 on Codabench. Zips: sbatchScriptEval.sh 9.
# srun --gres=gpu:1 python train.py --config configs/ablations/d15_full_train_j0_250ep.yaml --data "$DATA" --wandb --workers 4
# d16: d15 + j1's fused RoI pooling, 250 epochs on all of train (~12 h); vs d15 on Codabench. Zips: sbatchScriptEval.sh 9.
# srun --gres=gpu:1 python train.py --config configs/ablations/d16_full_train_j1_250ep.yaml --data "$DATA" --wandb --workers 4
# d17: d16 at 200 epochs (~9.5 h); vs d16 on Codabench. Zips: sbatchScriptEval.sh 9.
# srun --gres=gpu:1 python train.py --config configs/ablations/d17_full_train_j1_200ep.yaml --data "$DATA" --wandb --workers 4

# R: one change each on j1 (96 epochs, ~4 h), both compared with j1. Run them as a pair.
#   k1 small-object classes always copy-pasted (4 donors, 8 objects) | k2 RoI sampling 256 / 0.5 positives
# srun --gres=gpu:1 python train.py --config configs/ablations/k1_copy_paste_small.yaml --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/k2_roi_sparse.yaml       --data "$DATA" --wandb --workers 4

# S: localization (AP75) on j1, one change each, compared with j1 (k4 vs k5: GIoU share vs whole box-loss weight).
#   k3 IoU-aware ranking (h4 on j1) | k4 giou_weight 2.0 | k5 loss_box_reg 4.0
# srun --gres=gpu:1 python train.py --config configs/ablations/k3_iou_aware.yaml    --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/k4_giou_weight2.yaml --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/k5_box_loss4.yaml    --data "$DATA" --wandb --workers 4

# T: does j1's copy-paste help? Replaces k1 (stopped: -3.4 mAP / -4.5 AP_small vs j1 at step 14.8k). Compare with j1.
# srun --gres=gpu:1 python train.py --config configs/ablations/k6_no_copy_paste.yaml --data "$DATA" --wandb --workers 4

# U: alpha-GIoU (alpha 3) in place of the GIoU box term, one change on j1. Compare with j1 (and k4).
# srun --gres=gpu:1 python train.py --config configs/ablations/k7_alpha_giou.yaml --data "$DATA" --wandb --workers 4

# V: pre-fix results rerun on j1 (h2 / h6 / a2 were judged under the old mosaic that halved every object).
#   k8 patch-8 tokens at batch 32 (LR x sqrt 2; ~4-6x backbone compute, time one epoch; two changes vs j1)
#   k9 P2 heatmap loss (h6; STOPPED: mAP far below j1 mid-run) | k10 scale jitter 0.9-1.1 alone
# srun --gres=gpu:1 python train.py --config configs/ablations/k8_patch8_bs32.yaml   --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/k9_p2_heatmap.yaml    --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/k10_scale_jitter.yaml --data "$DATA" --wandb --workers 4

# W: half-pixel-correct RoIAlign (aligned=True) in j1's fused pooler, one change. Compare with j1 (AP75, AP_small).
# srun --gres=gpu:1 python train.py --config configs/ablations/k11_roi_aligned.yaml --data "$DATA" --wandb --workers 4
# Only if k11 beats j1: k11 + the P2 stem aligned with the ViT path. Compare with k11.
# srun --gres=gpu:1 python train.py --config configs/ablations/k12_roi_stem_aligned.yaml --data "$DATA" --wandb --workers 4

# X: exact 90-degree rotations alone, one change on j1. Compare with j1 (and k10, scale jitter alone).
# srun --gres=gpu:1 python train.py --config configs/ablations/k13_rot90.yaml --data "$DATA" --wandb --workers 4

# Y: bridge / playground recall (j1: AP .052 / .165; 40% / 26% never proposed, half of the lost bridges at proposal
#    IoU 0.3-0.5). One change each on j1, compare with j1 on per-class AP (bridge, playground) first. Run order:
#    pair 1: k11 (section W, aligned RoIAlign: 2 px = a near miss on 9 px boxes) + k14 (size-aware RPN positives)
#    pair 2: k15 (anchors on cell centres) + k16 (bridge / playground images sampled 2x)
# srun --gres=gpu:1 python train.py --config configs/ablations/k14_rpn_size_adaptive.yaml   --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/k15_rpn_centred_anchors.yaml --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/k16_oversample_hard.yaml     --data "$DATA" --wandb --workers 4
