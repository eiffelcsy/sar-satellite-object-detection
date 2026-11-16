@echo off
rem Train the final model from scratch (Windows; needs one NVIDIA GPU with ~20 GB, ~19.6 h on an L40S):
rem   scripts\train.bat              -^> runs\d16_final\model.pt
rem Sets up .venv (Python 3.13, requirements.txt) and downloads train\ into data\ on the first run. The pretrained
rem DINOv3 is gated: accept its licence at https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m and run
rem "hf auth login" once. If the run is interrupted, run the script again: it resumes from runs\d16_final\last.pt.
rem Extra arguments go to train.py (e.g. --epochs 1 --limit 64 for a quick check, --wandb to log to W&B).
setlocal
cd /d "%~dp0.."

if not exist .venv (
  py -3.13 -m venv .venv 2>nul || python -m venv .venv || exit /b 1
  .venv\Scripts\python -m pip install -r requirements.txt || exit /b 1
)
call .venv\Scripts\activate.bat

if not exist "data\train\instances.json" (
  hf download doem1997/cs701-sar-course-data --repo-type dataset --include train.tar.gz --local-dir data || exit /b 1
  tar -xzf data\train.tar.gz -C data || exit /b 1
)

python train.py --data data\train --resume %* || exit /b 1
echo done: runs\d16_final\model.pt (predict with: set WEIGHTS=runs\d16_final\model.pt ^& scripts\predict.bat)
