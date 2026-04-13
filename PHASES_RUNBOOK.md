# Runbook — Phases B (schema-stable) and C–F

This runbook assumes Phase A already produced:
- `artifacts/data_15min.parquet`

## 0) Activate venv

```powershell
Set-Location "E:\ML Based Prediction Air Pollutants"
.\.venv_std\Scripts\python.exe -V
```

## 1) (Recommended) Re-run Phase B with schema stabilization (write v3 artifacts)

This is the highest-ROI fix for improving RF/XGB: it prevents train/val/test column drift.

```powershell
.\.venv_std\Scripts\python.exe .\2_preprocess_and_features.py `
  --in-parquet  .\artifacts\data_15min.parquet `
  --out-train   .\artifacts\features_train_v3.parquet `
  --out-val     .\artifacts\features_val_v3.parquet `
  --out-test    .\artifacts\features_test_v3.parquet `
  --scaler-json .\artifacts\standard_scaler_v3.json `
  --schema-stable `
  --schema-json .\artifacts\feature_schema_v3.json `
  --log-level INFO
```

Optional (recommended if PM2.5 has extreme spikes):
- Add `--target-cap-quantile 0.995` to cap the target using a TRAIN-only quantile per station (leakage-safe).

Optional sanity check (fast):
```powershell
.\.venv_std\Scripts\python.exe .\tools\check_feature_splits.py `
  --train .\artifacts\features_train_v3.parquet `
  --val   .\artifacts\features_val_v3.parquet `
  --test  .\artifacts\features_test_v3.parquet `
  --out-json .\artifacts\feature_splits_report_v3.json
```

## 2) Phase C — Train classical models (Ridge / RF / XGB)

```powershell
.\.venv_std\Scripts\python.exe .\3_train_classical.py `
  --train .\artifacts\features_train_v3.parquet `
  --val   .\artifacts\features_val_v3.parquet `
  --test  .\artifacts\features_test_v3.parquet `
  --per-station-metrics `
  --log-level INFO
```

Optional upgrades:
- Tune XGBoost quickly on validation (8 configs):
  - add `--xgb-tune`
- Train per-station models (often boosts BHATAGAON without hurting others):
  - add `--per-station`
  - outputs `artifacts/classical_metrics_per_station.json` + `artifacts/model_*_{STATION}.*`

Speed knobs (optional):
- Keep Ridge full-data (default), but cap RF/XGB:
  - `--max-train-rows 50000 --xgb-max-train-rows 50000`
- Or train XGB on more rows:
  - `--xgb-max-train-rows 150000`

Outputs:
- `artifacts/model_ridge.pkl`
- `artifacts/model_rf.pkl`
- `artifacts/model_xgb.json`
- `artifacts/classical_metrics.json`

## 2b) (Recommended) Route the best model per station (validation-based)

After running `3_train_classical.py --per-station`, you can compute an overall score
using "best per station" routing (chosen by validation RMSE). This often boosts the
overall R² when one station behaves very differently (e.g., BHATAGAON).

```powershell
.\.venv_std\Scripts\python.exe .\9_route_models_by_station.py `
  --train .\artifacts\features_train_v3.parquet `
  --val   .\artifacts\features_val_v3.parquet `
  --test  .\artifacts\features_test_v3.parquet `
  --out-json .\artifacts\classical_metrics_routed.json `
  --log-level INFO
```

Outputs:
- `artifacts/classical_metrics_routed.json`

## 3) Phase D — LSTM (optional; only if time permits)

If you run LSTM, keep it lightweight due to the deadline and environment variability.

```powershell
.\.venv_std\Scripts\python.exe .\4_train_lstm.py --seq-len 96 --epochs 20
```

## 4) Phase E — Evaluate models

```powershell
.\.venv_std\Scripts\python.exe .\6_evaluate_models.py
```

## 5) Phase F — Plots

```powershell
.\.venv_std\Scripts\python.exe .\7_plots.py
```
