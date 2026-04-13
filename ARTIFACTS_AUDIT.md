# Artifacts Audit (Phase A & B)

## What exists now (confirmed by filesystem)
- `artifacts/data_15min.parquet` (~236.6 MB)
- `artifacts/ingest_run.log`
- `artifacts/standard_scaler.json`
- `artifacts/features_train.parquet` (~76.5 MB)
- `artifacts/features_val.parquet` (~60.4 MB)
- `artifacts/features_test.parquet` (~53.7 MB)

## What Phase B should guarantee (definition of “satisfactory”)

### Structure
- Columns include:
  - Identifiers: `station`, `timestamp`
  - Target: `PM2.5`
  - Temporal features: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `is_weekend`
  - Lag features per station for each base variable: `*_lag_{1,2,3,4,8,12,24,48,96}`
  - Rolling mean features leakage-safe: `*_rollmean_{4,12,24,48,96}`
- If `WD` exists in raw data: raw `WD` dropped, and `wd_sin`, `wd_cos` exist.

### Leakage safety
- Lags are created using `groupby('station').shift(k)` (no future information).
- Rolling means are created from `x.shift(1).rolling(w).mean()` (no current-step leakage).
- Standardization parameters (`standard_scaler.json`) are fitted **only on train** and applied to val/test.

### Cleanliness
- No duplicates on (`station`, `timestamp`) in final feature splits.
- No `NaN` remains in:
  - the target `PM2.5`
  - any newly created `PM2.5_lag_*` columns
- Chronological split per station:
  - earliest 70% train, next 15% val, last 15% test (no shuffle)

## Verification (fast, no re-running pipeline)

Run this audit script inside the same venv that produced the artifacts (must have pandas/pyarrow):

```powershell
Set-Location "E:\ML Based Prediction Air Pollutants"
.\.venv_clean\Scripts\python.exe .\tools\audit_artifacts.py > artifacts\audit_artifacts.json
```

Then inspect:
- `artifacts/audit_artifacts.json`

The key fields to check:
- `missing_required_cols` should be `[]`
- `nulls_in_target_and_pm25_lags` should be `0`
- `dup_station_timestamp` should be `0`
- `n_pm25_lag_cols` should be `9` (for lags 1..96)

