@echo off
rem Reproduce and score the final model's val and test results (Windows):
rem   scripts\predict.bat      -^> submission_val.zip, submission_test.zip, and their scores
rem Sets up .venv (Python 3.13, requirements.txt) and downloads val\, test\ and their released labels into data\ on
rem the first run. set WEIGHTS=path\to\model.pt overrides final_weights\model.pt; extra arguments go to predict.py.
setlocal
cd /d "%~dp0.."
if not defined WEIGHTS set "WEIGHTS=final_weights\model.pt"
if not exist "%WEIGHTS%" (echo error: %WEIGHTS% not found ^(put the submitted model.pt in final_weights\^) & exit /b 1)

if not exist .venv (
  py -3.13 -m venv .venv 2>nul || python -m venv .venv || exit /b 1
  .venv\Scripts\python -m pip install -r requirements.txt || exit /b 1
)
call .venv\Scripts\activate.bat

for %%S in (val test) do (
  if not exist "data\%%S_instances.json" (
    hf download doem1997/cs701-sar-course-data %%S.tar.gz %%S_labels.csv %%S_instances.json --repo-type dataset --local-dir data || exit /b 1
    tar -xzf "data\%%S.tar.gz" -C data || exit /b 1
  )
)

rem NMS IoU as in the Codabench submissions: 0.8 for val, 0.85 (configs\final.yaml) for test.
python predict.py --weights "%WEIGHTS%" --images-json data\val\images.json --out submission_val.zip --cache preprocess_cache\val --nms-thresh 0.8 %* || exit /b 1
python predict.py --weights "%WEIGHTS%" --images-json data\test\images.json --out submission_test.zip --cache preprocess_cache\test --nms-thresh 0.85 %* || exit /b 1
for %%S in (val test) do (
  python evaluate.py submission_%%S.zip --labels data\%%S_labels.csv --instances data\%%S_instances.json || exit /b 1
)
