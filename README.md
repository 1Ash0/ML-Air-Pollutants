# Air Quality Prediction (PM2.5) — Final Submission

This repository contains a complete, deadline-friendly pipeline for **PM2.5 prediction** from messy station Excel logs:

1. **Phase A** — Excel ingestion & reshape (`1_ingest_excel.py`)
2. **Phase B** — Cleaning + feature engineering (`2_preprocess_and_features.py`)
3. **Phase C** — Classical models + per-station routing (`3_train_classical.py`, `9_route_models_by_station.py`)
4. **Phase D** — LSTM (TensorFlow/Keras) (`4_train_lstm.py`)
5. **Phase E/F** — Final evaluation + report-ready plots (`5_evaluate_and_plot.py`)

## Final Results (Test)

Primary final system (recommended): **Per-station routed classical models**.
- Output: `artifacts/classical_metrics_routed_v4.json`
- Global test: **R² ≈ 0.9607**, RMSE ≈ 3.94 (PM2.5 units)

Deep learning baseline: LSTM.
- Output: `artifacts/lstm_metrics.json`
- Global test: **R² ≈ 0.8978**, RMSE ≈ 6.29

Unified comparison table across models (global + station):
- Output: `artifacts/metrics.csv`

Plots (dpi=300):
- `artifacts/plots/AIIMS_timeseries_3day_xgb_vs_lstm.png`
- `artifacts/plots/scatter_xgb_vs_lstm.png`
- `artifacts/plots/model_comparison_bars.png`

## How to Reproduce (high level)

See `PHASES_RUNBOOK.md` for the exact PowerShell commands used to regenerate the v4 features, train models, and generate the report plots.

## Dependencies

- Ingestion requirements: `requirements_ingest.txt`
- ML requirements: `requirements_ml.txt`

## Notes on GitHub size

Large datasets / parquet splits / model binaries are intentionally ignored by `.gitignore`.
Push only the code + small report outputs unless you explicitly set up Git LFS.

