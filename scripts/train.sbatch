#!/bin/bash
# Train the final model on one GPU with SLURM (~19.6 h and 19.7 GB peak on an NVIDIA L40S).
#   DATA=/path/to/cs701-sar-course-data/train sbatch scripts/train.sbatch
# If the job hits the time limit, submit the same command again: --resume continues from runs/d16_final/last.pt.
# Set --partition / --account / --qos to your cluster's values.
#SBATCH --job-name=sar-train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=%u.%j.out
#SBATCH --partition=student
#SBATCH --account=student
#SBATCH --qos=studentqos

set -euo pipefail
source .venv/bin/activate
srun --gres=gpu:1 python train.py --data "${DATA:?set DATA to the course data train/ dir}" --resume "$@"
