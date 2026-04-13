# Submission Checklist (GitHub)

## What to include

- Code:
  - `1_ingest_excel.py`
  - `2_preprocess_and_features.py`
  - `3_train_classical.py`
  - `4_train_lstm.py`
  - `5_evaluate_and_plot.py`
  - `9_route_models_by_station.py`
- Docs:
  - `README.md`
  - `PROJECT_BLUEPRINT.md`
  - `PHASES_RUNBOOK.md`
  - `IMPLEMENTATION_STATUS.md`
- Requirements:
  - `requirements_ingest.txt`
  - `requirements_ml.txt`
- Final outputs (small):
  - `artifacts/metrics.csv`
  - `artifacts/classical_metrics_routed_v4.json`
  - `artifacts/classical_metrics_per_station.json`
  - `artifacts/lstm_metrics.json`
  - `artifacts/plots/*.png`

## What NOT to include (too large)

- Raw Excel files: `Dataset/`
- Parquet splits: `artifacts/*v*.parquet`, `artifacts/data_15min.parquet`
- Model binaries: `*.pkl`, `*.keras`, station RF pickles (hundreds of MB)

Use Git LFS only if your instructor explicitly requires pushing model binaries.

## One-command packaging (recommended)

```powershell
Set-Location "E:\ML Based Prediction Air Pollutants"
powershell -ExecutionPolicy Bypass -File ".\tools\package_submission.ps1"
```

This creates a timestamped folder under `artifacts/` with everything you should upload.

