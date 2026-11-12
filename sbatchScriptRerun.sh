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
#SBATCH --job-name=EiffelRerun     # Give the job a name

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Runs the clean-ablation and re-run configs, one after another, each on held-out data (val_fraction 0.1) with W&B.
#
#   sbatch sbatchScriptRerun.sh              # the DEFAULT_RUNS below
#   sbatch sbatchScriptRerun.sh l1 k20       # these run IDs instead (one job each is safer: see time below)
#   DRY_RUN=1 bash sbatchScriptRerun.sh l1   # print what would run, train nothing
#
# A run ID is the file-name prefix of its config: l1 -> configs/ablations/l1_*.yaml. To add a run, write its config
# (base: the parent, list only the change), add one line to the registry below, and, to run it by default, add its ID
# to DEFAULT_RUNS. Any existing config works as an ID too, e.g. k20 or k1 to re-run a run that was stopped early.
#
# Time: runs go in sequence inside one job. One 96-epoch run on j1's recipe takes ~7 h on an L40S, so at most three fit
# the 24 h limit; submit longer or more runs as separate jobs.
#
# Resume: every run is started with --resume. If the job hits the time limit or crashes, submit the same command again:
# a run with a last.pt checkpoint continues after its last finished epoch and the same W&B run. A run whose model.pt
# already exists is skipped, so a repeat submission never retrains a finished run.
#
# Registry (ID: what it tests, parent, what to compare):
#   l1  input ablation: pseudo-RGB vs amplitude repeated 3x. Parent j1_roi_fused (held-out mAP 46.34, AP75 49.03,
#       macro-F1 96.28). Only preprocess.mode changes (despeckle -> amplitude). ~7 h. Closes the AB-02 gap.
#   l2  DINOv3's training-mode RoPE coordinate rescale off (backbone.params.rope_augment: false), so train and eval
#       tokens match. Parent j1_roi_fused; compare at the same epoch (j1 mAP: 30.4 at 16, 35.9 at 24, 42.9 at 48,
#       46.3 at 96). ~7 h.
#   l3  Cascade R-CNN as published on the mosaic-free j1: 3 x 384-wide heads, IoU 0.5/0.6/0.7, stage scores averaged
#       at inference. Parent j1_roi_fused; compare AP75 and per-class AP. 39.87 M trainable. Slower than j1 (three
#       RoI passes): if the job hits 24 h, resubmit.
#   (re-runs of stopped runs and further clean ablations: add them here)
DEFAULT_RUNS=(l1 l2 l3)

# This command assumes that you've already created the environment previously
# We're using an absolute path here. You may use a relative path, as long as SRUN is execute in the same working directory
source .venv/bin/activate

DATA=/common/scratch/users/e/eiffelchong.2023/cs701-sar-course-data/train/

RUNS=("$@")
[ ${#RUNS[@]} -eq 0 ] && RUNS=("${DEFAULT_RUNS[@]}")

for ID in "${RUNS[@]}"; do
  CONFIGS=(configs/ablations/${ID}_*.yaml)
  if [ ${#CONFIGS[@]} -ne 1 ] || [ ! -f "${CONFIGS[0]}" ]; then
    echo "[$ID] expected exactly one configs/ablations/${ID}_*.yaml, found: ${CONFIGS[*]}; skipping"
    continue
  fi
  CONFIG="${CONFIGS[0]}"
  NAME=$(python -c "from sarbench.config import load_config; print(load_config('$CONFIG')['name'])")
  if [ -f "runs/$NAME/model.pt" ]; then
    echo "[$ID] runs/$NAME/model.pt exists: already finished, skipping"
    continue
  fi
  echo "[$ID] $CONFIG -> runs/$NAME ($(date))"
  if [ -n "$DRY_RUN" ]; then
    echo "  srun --gres=gpu:1 python train.py --config $CONFIG --data $DATA --wandb --workers 4 --resume"
    continue
  fi
  srun --gres=gpu:1 python train.py --config "$CONFIG" --data "$DATA" --wandb --workers 4 --resume
done
