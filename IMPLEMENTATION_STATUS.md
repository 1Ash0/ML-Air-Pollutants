# Implementation Status (as of 2026-04-13)

## What's implemented

### Phase A — Excel ingestion / reshape
- Script: `1_ingest_excel.py`
- Outputs:
  - `artifacts/_ingest_chunks/*.parquet` (streamed)
  - `artifacts/data_15min.parquet` (merged tidy 15-min table)
- Key logic:
  - Recursively scans `Dataset/` for `.xls/.xlsx`
  - Per sheet: reads a small preview to locate `Date & Time` header row, renames value columns from that row, and skips the units row
  - Parses timestamps with both `DD-MM-YYYY HH:MM` and `DD/MM/YYYY HH:MM`
  - Cleans status tokens (maint/calib/power/link/warm/ref/fail/off + null-ish strings) to NaN and coerces to numeric
  - Uses chunk streaming to avoid RAM blowups; logs to console and `artifacts/ingest_run.log`
- Fix applied:
  - Resolved pandas crash: `AttributeError: 'Series' object has no attribute '_hasna'` by removing fragile inplace boolean assignment in `_last_nonblank_col_index` and avoiding regex replace paths that trigger dtype edge cases.

### Phase B — Cleaning + normalization + feature engineering
- Script: `2_preprocess_and_features.py`
- Outputs:
  - `artifacts/features_train*.parquet`, `artifacts/features_val*.parquet`, `artifacts/features_test*.parquet`
  - `artifacts/standard_scaler*.json`
- Key logic:
  - Strict 15-min grid per station + duplicate timestamp aggregation by mean
  - Linear interpolation limited to 4 forward steps (<= 1 hour)
  - Sentinel cleaning: negative / placeholder values (e.g., `-2e9`) -> NaN before splitting
  - Temporal features: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `is_weekend`
  - WD handling: `WD` -> `wd_sin`, `wd_cos`, drop raw `WD`
  - Lags per station: `[1,2,3,4,8,12,24,48,96]`
  - Rolling means leakage-safe: `x.shift(1).rolling(w).mean()` with windows `[4,12,24,48,96]`
  - Leakage-safe per-station split 70/15/15 (no shuffle) with history tail for val/test feature computation
- Recent stability upgrades (implemented now):
  - `build_features_with_context(..., base_cols=...)` to prevent split-by-split schema drift
  - `--schema-stable` option to force train/val/test to share the exact same columns (train-based), writing `--schema-json` (default: `artifacts/feature_schema.json`)

### Phase C — Classical models (Ridge / RF / XGBoost)
- Script: `3_train_classical.py`
- Outputs:
  - `artifacts/model_ridge.pkl`, `artifacts/model_rf.pkl`, `artifacts/model_xgb.json`
  - `artifacts/classical_metrics.json`
- Key logic:
  - Train/val/test feature alignment (train-based)
  - Ridge pipeline: `SimpleImputer(median)` -> `StandardScaler` -> `Ridge`, alpha tuned on val
  - RF pipeline: `SimpleImputer(median)` -> `RandomForestRegressor(300, max_depth=20, random_state=42)` with `n_jobs=-1` fallback to 1 if restricted
  - XGB: compatibility handling for `early_stopping_rounds` placement across xgboost versions
  - New knobs:
    - `--max-train-rows`, `--xgb-max-train-rows`, `--ridge-max-rows`
    - `--per-station-metrics` to add per-station test metrics
    - `--xgb-tune` small validation sweep (8 configs)
    - `--per-station` trains separate models per station (writes `artifacts/classical_metrics_per_station.json`)

### Diagnostics / helpers
- Script: `8_xgb_sweep.py` (XGBoost config sweep using `xgboost.train` + DMatrix + early stopping)
- Script: `tools/audit_artifacts.py` (artifact sanity summary)
- Script: `tools/check_feature_splits.py` (schema diff + target sanity across splits)

### Phase D — LSTM (TensorFlow/Keras)
- Script: `4_train_lstm.py`
- Outputs:
  - `artifacts/model_lstm.keras`
  - `artifacts/lstm_scalers.pkl`
  - `artifacts/lstm_metrics.json`
- Key logic:
  - Fits `StandardScaler` on TRAIN features only, applies to val/test
  - Fits separate `StandardScaler` on TRAIN target only (neural stability)
  - Station-safe sliding windows (no sequence crosses stations), sequence length default T=96
  - Minimal architecture: `LSTM(32)` -> `Dense(16)` -> `Dense(1)` with early stopping
  - Logs tensor shapes via `element_spec` and prints `model.summary()` into logs

## What has been executed (per your logs)
- Phase A completed: merged `artifacts/data_15min.parquet` exists.
- Phase B completed: v2 splits were generated after sentinel cleaning fixes.
- Phase C completed: Ridge/RF/XGB produced `artifacts/classical_metrics.json` on v2 splits.

## Next action (highest ROI)
1. Re-run Phase B once with `--schema-stable` and write a new set of features (e.g., v3).
2. Retrain Phase C on the stabilized features and compare metrics (especially XGB and RF).
