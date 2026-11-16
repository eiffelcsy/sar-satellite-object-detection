#!/usr/bin/env bash
# Reproduce and score the final model's val and test results (macOS / Linux):
#   bash scripts/predict.sh      -> submission_val.zip, submission_test.zip, and their macro-F1, mAP and Δm
# Sets up .venv (Python 3.13, requirements.txt) and downloads val/, test/ and their released labels into data/ on the
# first run. WEIGHTS=path/to/model.pt overrides final_weights/model.pt; extra arguments go to predict.py.
set -euo pipefail
cd "$(dirname "$0")/.."
WEIGHTS=${WEIGHTS:-final_weights/model.pt}
[ -f "$WEIGHTS" ] || { echo "error: $WEIGHTS not found (put the submitted model.pt in final_weights/)"; exit 1; }

if [ ! -d .venv ]; then
  PYTHON=${PYTHON:-$(command -v python3.13 || command -v python3)}
  "$PYTHON" -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi
source .venv/bin/activate

for SPLIT in val test; do
  if [ ! -f "data/$SPLIT/images.json" ] || [ ! -f "data/${SPLIT}_instances.json" ]; then
    hf download doem1997/cs701-sar-course-data "$SPLIT.tar.gz" "${SPLIT}_labels.csv" "${SPLIT}_instances.json" \
      --repo-type dataset --local-dir data
    tar -xzf "data/$SPLIT.tar.gz" -C data
  fi
done

# NMS IoU as in the Codabench submissions: 0.8 for val, 0.85 (configs/final.yaml) for test.
for SPLIT in val test; do
  NMS=$([ "$SPLIT" = val ] && echo 0.8 || echo 0.85)
  python predict.py --weights "$WEIGHTS" --images-json "data/$SPLIT/images.json" --out "submission_$SPLIT.zip" \
    --cache "preprocess_cache/$SPLIT" --nms-thresh "$NMS" "$@"
done
for SPLIT in val test; do
  python evaluate.py "submission_$SPLIT.zip" --labels "data/${SPLIT}_labels.csv" --instances "data/${SPLIT}_instances.json"
done
