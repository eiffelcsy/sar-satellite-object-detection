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
srun --gres=gpu:1 python train.py --config configs/ablations/b1_full_blocks.yaml  --data "$DATA" --wandb --workers 4
srun --gres=gpu:1 python train.py --config configs/ablations/b2_rank_early.yaml   --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/b3_rank_uniform.yaml --data "$DATA" --wandb --workers 4
# srun --gres=gpu:1 python train.py --config configs/ablations/b4_convpass.yaml     --data "$DATA" --wandb --workers 4
srun --gres=gpu:1 python train.py --config configs/ablations/b5_hybrid.yaml       --data "$DATA" --wandb --workers 4
