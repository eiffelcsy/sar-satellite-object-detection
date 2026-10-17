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
ROOT=/common/scratch/users/e/eiffelchong.2023/cs701-sar-course-data
DATA="$ROOT/train"
CONFIG=configs/ablations/e4_stem_120ep.yaml
MODEL=runs/giou_e4_stem_120ep/model.pt

# Evaluation / submission (no training; reuses the trained checkpoint):
# 0) Post-processing sweep on the held-out train split: score threshold x NMS IoU, no retraining. Prints the best
#    setting and writes postprocess_sweep.json next to the model; apply it to 2) / 3) with --score-thresh and
#    --nms-thresh (e.g. --score-thresh 0.001 --nms-thresh 0.6).
srun --gres=gpu:1 python sweep_postprocess.py --config "$CONFIG" --load "$MODEL" --data "$DATA" --workers 4

# 1) Held-out train split (labelled): val metrics + submission_val.zip
# srun --gres=gpu:1 python train.py --config "$CONFIG" --load "$MODEL" \
#   --data "$DATA" --out runs/eval_heldout

# 2) Official val split (unlabelled val/images.json): predictions only -> submission_val.zip
srun --gres=gpu:1 python train.py --config "$CONFIG" --load "$MODEL" \
  --score-thresh 0.001 --nms-thresh 0.6 \
  --predict "$ROOT/val/images.json" --predict-split val --out runs/eval_val/ \
  --preprocess-cache preprocess_cache/val

# 3) Official test split (unlabelled test/images.json): predictions only -> submission_test.zip
# srun --gres=gpu:1 python train.py --config "$CONFIG" --load "$MODEL" \
#   --predict "$ROOT/test/images.json" --predict-split test --out runs/eval_test