# Runbook (v4) - Reproduce Final Results

This runbook assumes Phase A already produced:
- `artifacts/data_15min.parquet`

All commands below are PowerShell.

## 0) Activate venv

```powershell
Set-Location "E:\ML Based Prediction Air Pollutants"
.\.venv_std\Scripts\python.exe -V
```

## 1) Phase B (v4 features) - schema-stable + leakage-safe target capping

```powershell
.\.venv_std\Scripts\python.exe ".\2_preprocess_and_features.py" `
  --in-parquet ".\artifacts\data_15min.parquet" `
  --out-train ".\artifacts\features_train_v4.parquet" `
  --out-val ".\artifacts\features_val_v4.parquet" `
  --out-test ".\artifacts\features_test_v4.parquet" `
  --scaler-json ".\artifacts\standard_scaler_v4.json" `
  --schema-stable `
  --schema-json ".\artifacts\feature_schema_v4.json" `
  --target-cap-quantile 0.995 `
  --log-level INFO
```

Sanity check:
```powershell
.\.venv_std\Scripts\python.exe ".\tools\check_feature_splits.py" `
  --train ".\artifacts\features_train_v4.parquet" `
  --val ".\artifacts\features_val_v4.parquet" `
  --test ".\artifacts\features_test_v4.parquet" `
  --out-json ".\artifacts\feature_splits_report_v4.json"
```

## 2) Phase C (per-station classical training) + routing (final system)

Train per-station Ridge/RF/XGB and tune XGB quickly (8 configs per station):
```powershell
.\.venv_std\Scripts\python.exe ".\3_train_classical.py" `
  --train ".\artifacts\features_train_v4.parquet" `
  --val ".\artifacts\features_val_v4.parquet" `
  --test ".\artifacts\features_test_v4.parquet" `
  --per-station `
  --xgb-tune `
  --log-level INFO
```

Route the best model per station by validation RMSE and evaluate on test:
```powershell
.\.venv_std\Scripts\python.exe ".\9_route_models_by_station.py" `
  --train ".\artifacts\features_train_v4.parquet" `
  --val ".\artifacts\features_val_v4.parquet" `
  --test ".\artifacts\features_test_v4.parquet" `
  --out-json ".\artifacts\classical_metrics_routed_v4.json" `
  --log-level INFO
```

## 3) Phase D (LSTM baseline)

```powershell
.\.venv_std\Scripts\python.exe ".\4_train_lstm.py" `
  --train ".\artifacts\features_train_v4.parquet" `
  --val ".\artifacts\features_val_v4.parquet" `
  --test ".\artifacts\features_test_v4.parquet" `
  --seq-len 96 `
  --batch-size 128 `
  --epochs 20 `
  --reduce-lr-on-plateau `
  --log-level INFO
```

## 4) Phase E/F (final evaluation + plots)

```powershell
.\.venv_std\Scripts\python.exe ".\5_evaluate_and_plot.py" `
  --test ".\artifacts\features_test_v4.parquet" `
  --artifacts-dir ".\artifacts" `
  --lstm-model ".\artifacts\model_lstm.keras" `
  --lstm-scalers ".\artifacts\lstm_scalers.pkl" `
  --out-metrics ".\artifacts\metrics.csv" `
  --out-plots ".\artifacts\plots" `
  --dpi 300 `
  --log-level INFO
```

## 5) Viva pack (recommended)

Generate 20+ viva-ready plots (correlation, station differences, model performance, routing justification).

Note: `features_*_v4.parquet` files are large and may be stored under `archive/` after GitHub cleanup.
This script auto-finds archived copies if the `artifacts/` paths are missing.

```powershell
.\.venv_std\Scripts\python.exe ".\6_viva_plots.py" `
  --log-level INFO
```

## 6) (NEW) Multi-pollutant forecasting (sir feedback)

Train a *single* multi-output model that predicts multiple pollutants **15 minutes ahead** (`t+1` for 15-min data).
This produces `artifacts/multioutput_metrics.csv` and `artifacts/multioutput_meta.json`.

```powershell
.\.venv_std\Scripts\python.exe ".\3_train_multioutput.py" `
  --train ".\artifacts\features_train_v4.parquet" `
  --val   ".\artifacts\features_val_v4.parquet" `
  --test  ".\artifacts\features_test_v4.parquet" `
  --horizon-steps 1 `
  --skip-xgb `
  --log-level INFO
```

Create plots for the multi-output results:
```powershell
.\.venv_std\Scripts\python.exe ".\7_multioutput_plots.py" `
  --metrics-csv ".\artifacts\multioutput_metrics.csv" `
  --log-level INFO
```

Route best model per (station, target) using validation RMSE (multi-output routing):
```powershell
.\.venv_std\Scripts\python.exe ".\10_route_multioutput_by_station_target.py" `
  --train ".\artifacts\features_train_v4.parquet" `
  --val   ".\artifacts\features_val_v4.parquet" `
  --test  ".\artifacts\features_test_v4.parquet" `
  --out-json ".\artifacts\multioutput_routed.json" `
  --out-csv  ".\artifacts\multioutput_metrics_routed.csv" `
  --log-level INFO
```

Then re-run the multi-output plots (adds routed visuals if routed csv exists):
```powershell
.\.venv_std\Scripts\python.exe ".\7_multioutput_plots.py" `
  --metrics-csv ".\artifacts\multioutput_metrics.csv" `
  --routed-csv  ".\artifacts\multioutput_metrics_routed.csv" `
  --log-level INFO
```

Train a multi-output LSTM (optional, slower but good for viva comparison):
```powershell
.\.venv_std\Scripts\python.exe ".\4_train_lstm_multioutput.py" `
  --train ".\artifacts\features_train_v4.parquet" `
  --val   ".\artifacts\features_val_v4.parquet" `
  --test  ".\artifacts\features_test_v4.parquet" `
  --horizon-steps 1 `
  --seq-len 96 `
  --batch-size 128 `
  --epochs 20 `
  --reduce-lr-on-plateau `
  --log-level INFO
```

## 7) (NEW) Comparative study: Global vs Per-station vs Routed

To address the "per-station may overfit" viva question, train **global** (all-stations) models and write a comparison JSON.

```powershell
.\.venv_std\Scripts\python.exe ".\8_compare_global_vs_station.py" `
  --train ".\artifacts\features_train_v4.parquet" `
  --val   ".\artifacts\features_val_v4.parquet" `
  --test  ".\artifacts\features_test_v4.parquet" `
  --out-json ".\artifacts\pm25_comparison.json" `
  --log-level INFO
```
