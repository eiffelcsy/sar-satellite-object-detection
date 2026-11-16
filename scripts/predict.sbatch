#!/bin/bash
# Write and score the val and test submissions of the trained model (~25 min for both, mostly SAR-BM3D on the CPU workers).
#   DATA=/path/to/cs701-sar-course-data sbatch scripts/predict.sbatch
# DATA holds val/, test/ and the released {val,test}_labels.csv and {val,test}_instances.json.
# Set --partition / --account / --qos to your cluster's values.
#SBATCH --job-name=sar-predict
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=%u.%j.out
#SBATCH --partition=student
#SBATCH --account=student
#SBATCH --qos=studentqos

set -euo pipefail
source .venv/bin/activate
: "${DATA:?set DATA to the course data root (with val/ and test/)}"
WEIGHTS=${WEIGHTS:-final_weights/model.pt}
for SPLIT in val test; do
  NMS=$([ "$SPLIT" = val ] && echo 0.8 || echo 0.85)  # as in the Codabench submissions
  srun --gres=gpu:1 python predict.py --weights "$WEIGHTS" --images-json "$DATA/$SPLIT/images.json" \
    --out "submission_$SPLIT.zip" --cache "preprocess_cache/$SPLIT" --nms-thresh "$NMS"
  python evaluate.py "submission_$SPLIT.zip" --labels "$DATA/${SPLIT}_labels.csv" \
    --instances "$DATA/${SPLIT}_instances.json"
done
