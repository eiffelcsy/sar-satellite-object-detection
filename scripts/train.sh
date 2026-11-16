#!/usr/bin/env bash
# Train the final model from scratch (macOS / Linux; needs one NVIDIA GPU with ~20 GB, ~19.6 h on an L40S):
#   bash scripts/train.sh              -> runs/d16_final/model.pt
# Sets up .venv (Python 3.13, requirements.txt) and downloads train/ into data/ on the first run. The pretrained
# DINOv3 is gated: accept its licence at https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m and run
# `hf auth login` once. If the run is interrupted, run the script again: it resumes from runs/d16_final/last.pt.
# Extra arguments go to train.py (e.g. --epochs 1 --limit 64 for a quick check, --wandb to log to W&B).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  PYTHON=${PYTHON:-$(command -v python3.13 || command -v python3)}
  "$PYTHON" -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi
source .venv/bin/activate

if [ ! -f data/train/instances.json ]; then
  hf download doem1997/cs701-sar-course-data --repo-type dataset --include train.tar.gz --local-dir data
  tar -xzf data/train.tar.gz -C data
fi

python train.py --data data/train --resume "$@"
echo "done: runs/d16_final/model.pt (predict with: WEIGHTS=runs/d16_final/model.pt bash scripts/predict.sh)"
