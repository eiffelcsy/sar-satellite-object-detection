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
srun --gres=gpu:1 python train.py --config configs/ablations/e4_stem_120ep.yaml  --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/e5_stem_150ep.yaml  --data "$DATA" --wandb --workers 4

# F: the final model, after the e* results: point configs/ablations/d10_full_train.yaml's `base` at the winning
# recipe (e4 or e5), train on all of train (no held-out metrics), then make the Codabench zips with
# sbatchScriptEval.sh 2) / 3) using --score-thresh 0.001 --nms-thresh 0.6 (the d1 post-processing sweep's best).
# srun --gres=gpu:1 python train.py --config configs/ablations/d10_full_train.yaml    --data "$DATA" --wandb --workers 4
