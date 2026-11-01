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
# RF-DETR has its own environment (it needs transformers >= 5.1 and PyTorch Lightning; keep it apart from .venv).
# One-time setup on the login node (it has internet; compute nodes may not):
#   python3 -m venv .venv-rfdetr && source .venv-rfdetr/bin/activate
#   pip install "rfdetr[train]==1.11.2"
#   python -c "from rfdetr import RFDETRSmall, RFDETRMedium; RFDETRSmall(); RFDETRMedium()"   # caches the weights
# .venv-rfdetr was built with this module's Python (it links libpython3.12.so), so load it first.
module load Python/3.12.8-GCCcore-13.3.0
source .venv-rfdetr/bin/activate

ROOT=/common/scratch/users/e/eiffelchong.2023/cs701-sar-course-data
DATA="$ROOT/train"

# R1) Export the course data in RF-DETR's layout, same held-out split as train.py (CPU only, under a minute for
#     raw, which links the images; amp_log_edge writes no-BM3D pseudo-RGB PNGs). It must run before R2; it skips
#     files that already exist, so leaving it on is harmless.
srun python rfdetr_track/export_dataset.py --data "$DATA" --out rfdetr_data/heldout_raw
# srun python rfdetr_track/export_dataset.py --data "$DATA" --out rfdetr_data/full_raw --val-fraction 0

# R2) RF-DETR Small (DINOv2 ViT-S windowed, 512 px native, COCO-pretrained, every weight trained) on the held-out
#     split. --resume: resubmit unchanged after a time-limit kill (continues from last.ckpt).
# srun --gres=gpu:1 python rfdetr_track/train_rfdetr.py --dataset rfdetr_data/heldout_raw \
#   --output runs/rfdetr_small_raw --variant small --epochs 60 --batch-size 16 --workers 4 --resume --wandb

# R2b) Extend R2 (60 epochs, all at full LR: RF-DETR's default drop is at epoch 100) by 25 epochs to 85, with the
#      LR 10x lower from epoch 70 (10 more at full LR, then 15 low): continues from runs/rfdetr_small_raw/last.ckpt
#      (optimizer, EMA, epoch 60) and keeps updating checkpoint_best_*.pth there. Then rerun sbatchScriptEval.sh
#      section 16. Stops if last.ckpt is missing (--resume would otherwise train 85 epochs from scratch).
test -f runs/rfdetr_small_raw/last.ckpt || { echo "runs/rfdetr_small_raw/last.ckpt not found: nothing to extend"; exit 1; }
srun --gres=gpu:1 python rfdetr_track/train_rfdetr.py --dataset rfdetr_data/heldout_raw \
  --output runs/rfdetr_small_raw --variant small --epochs 85 --lr-drop 70 --batch-size 16 --workers 4 --resume --wandb

# R3) Held-out metrics, comparable with the Faster R-CNN runs (j1 .4758 / k19 .4817 mAP, per-class AP). Try
#     --single-class (all boxes take the image class) too.
# srun --gres=gpu:1 python rfdetr_track/predict_rfdetr.py --checkpoint runs/rfdetr_small_raw/checkpoint_best_ema.pth \
#   --data "$DATA" --out runs/rfdetr_small_raw/heldout --threshold 0.001

# R4) Submission zips (official val / test images).
# srun --gres=gpu:1 python rfdetr_track/predict_rfdetr.py --checkpoint runs/rfdetr_small_raw/checkpoint_best_ema.pth \
#   --images-json "$ROOT/val/images.json" --split val --out runs/rfdetr_small_raw/val
# srun --gres=gpu:1 python rfdetr_track/predict_rfdetr.py --checkpoint runs/rfdetr_small_raw/checkpoint_best_ema.pth \
#   --images-json "$ROOT/test/images.json" --split test --out runs/rfdetr_small_raw/test

# R5) If R3 beats k19: the larger variant, then the final model on all of train (export full_raw first).
# srun --gres=gpu:1 python rfdetr_track/train_rfdetr.py --dataset rfdetr_data/heldout_raw \
#   --output runs/rfdetr_medium_raw --variant medium --epochs 60 --batch-size 16 --workers 4 --resume --wandb
# srun --gres=gpu:1 python rfdetr_track/train_rfdetr.py --dataset rfdetr_data/full_raw \
#   --output runs/rfdetr_small_full --variant small --epochs 60 --batch-size 16 --workers 4 --resume --wandb
