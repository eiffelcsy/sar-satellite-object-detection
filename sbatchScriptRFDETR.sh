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
# srun python rfdetr_track/export_dataset.py --data "$DATA" --out rfdetr_data/heldout_raw
# srun python rfdetr_track/export_dataset.py --data "$DATA" --out rfdetr_data/full_raw --val-fraction 0

# R2) RF-DETR Small (DINOv2 ViT-S windowed, 512 px native, COCO-pretrained, every weight trained) on the held-out
#     split. --resume: resubmit unchanged after a time-limit kill (continues from last.ckpt).
# srun --gres=gpu:1 python rfdetr_track/train_rfdetr.py --dataset rfdetr_data/heldout_raw \
#   --output runs/rfdetr_small_raw --variant small --epochs 60 --batch-size 16 --workers 4 --resume --wandb

# R2b) Extend R2 (60 epochs, all at full LR: RF-DETR's default drop is at epoch 100) by 25 epochs to 85, with the
#      LR 10x lower from epoch 70 (10 more at full LR, then 15 low): continues from runs/rfdetr_small_raw/last.ckpt
#      (optimizer, EMA, epoch 60) and keeps updating checkpoint_best_*.pth there. Then rerun sbatchScriptEval.sh
#      section 16. Stops if last.ckpt is missing (--resume would otherwise train 85 epochs from scratch).
# test -f runs/rfdetr_small_raw/last.ckpt || { echo "runs/rfdetr_small_raw/last.ckpt not found: nothing to extend"; exit 1; }
# srun --gres=gpu:1 python rfdetr_track/train_rfdetr.py --dataset rfdetr_data/heldout_raw \
#   --output runs/rfdetr_small_raw --variant small --epochs 80 --lr-drop 65 --batch-size 16 --workers 4 --resume --wandb

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

# R6) RF-DETR Large (2026: ViT-S windowed, 4 decoder layers, 33.9M, every weight trained) at 512 px (native 704; the
#     positional encoding is resized), on the no-BM3D pseudo-RGB input (amplitude, log-amplitude, Sobel edges) with
#     k17's hard-class oversampling: every train image once, bridge / playground images twice (~10.6k images per
#     epoch, +26% time). Schedule: 200 epochs, LR 1e-4 / encoder 1.5e-4 (1 warmup epoch) until epoch 162, then 10x
#     lower for the last 38: the Small run's proportion (drop at 65 of 80). ~8-11 h in one job (Small: 100 min for 60
#     epochs); if a job is cut off, resubmit unchanged and --resume continues. Small at 80 epochs (drop at 65): mAP .4482, bridge .035, playground .091.
#     Large's COCO weights are not cached yet: on the login node, once:
#       python -c "from rfdetr import RFDETRLarge; RFDETRLarge()"
#     a) Export (CPU; writes ~10.4k PNGs for amp_log_edge, the repeats are links; skips files that exist).
# srun python rfdetr_track/export_dataset.py --data "$DATA" --out rfdetr_data/heldout_ale_repeat \
#   --input amp_log_edge --repeat bridge=2 playground=2
#     b) Train. --resume: resubmit unchanged after a time-limit kill (continues from last.ckpt).
# srun --gres=gpu:1 python rfdetr_track/train_rfdetr.py --dataset rfdetr_data/heldout_ale_repeat \
#   --output runs/rfdetr_large_ale_repeat --variant large --epochs 200 --lr-drop 162 --batch-size 16 --workers 4 \
#   --resume --wandb

# R7-R9) Faster R-CNN-track lessons on top of R6 (rfdetr_track/fusion.py), same data and schedule, one change each:
#     R7: + a frozen DINOv3 ViT-B/16 (the Faster R-CNN backbone, ~86M frozen) fused into the projector output by
#         zero-init 1x1 convs (training starts exactly at R6's pretrained RF-DETR), + an image classifier on its
#         [CLS, mean patch] tokens (loss weight 0.1, as the Faster R-CNN classifier); predict_rfdetr.py then takes the
#         image label from it (--label-mode cls) for macro-F1 and the other-class box prior. ~34M trainable, ~120M total.
#     R8: R7 + DoRA r16 (attn + mlp) and Convpass 64 on every DINOv3 block (~4.4M): the frozen features adapted to
#         SAR, as in the Faster R-CNN track. ~38M trainable, ~124M total.
#     R9: R8 + a stride-8 P3 level (cars / ships / aircraft; Faster R-CNN's P2 detail stem helped small objects).
#         P3 costs ~2.7M, so DoRA drops to r8 to stay under 40M. The pretrained P4-only weights are remapped (the
#         decoder starts reading mostly P4). ~39.6M trainable, ~126M total.
#     The new parameters train at --lr (1e-4). train_rfdetr.py prints the counts and stops if over budget. The ViT-B
#     forward (R7) and its backward (R8 / R9) add time, so R8 / R9 may need a second job: resubmit unchanged
#     (--resume). They use R6's export (R6a), so run that first.
#     The DINOv3 ViT-B weights come from Hugging Face (gated; the Faster R-CNN track already caches them).
srun --gres=gpu:1 python rfdetr_track/train_rfdetr.py --dataset rfdetr_data/heldout_ale_repeat \
  --output runs/rfdetr_large_ale_repeat_dino --variant large --epochs 200 --lr-drop 162 --batch-size 16 --workers 4 \
  --dino --image-cls-weight 0.1 --resume --wandb
srun --gres=gpu:1 python rfdetr_track/train_rfdetr.py --dataset rfdetr_data/heldout_ale_repeat \
  --output runs/rfdetr_large_ale_repeat_dino_dora --variant large --epochs 200 --lr-drop 162 --batch-size 16 \
  --workers 4 --dino --dino-dora-rank 16 --dino-convpass-dim 64 --image-cls-weight 0.1 --resume --wandb
srun --gres=gpu:1 python rfdetr_track/train_rfdetr.py --dataset rfdetr_data/heldout_ale_repeat \
  --output runs/rfdetr_large_ale_repeat_dino_dora_p3 --variant large --epochs 200 --lr-drop 162 --batch-size 16 \
  --workers 4 --dino --dino-dora-rank 8 --dino-convpass-dim 64 --image-cls-weight 0.1 --p3 --resume --wandb
