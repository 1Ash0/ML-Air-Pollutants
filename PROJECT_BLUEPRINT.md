# Air Quality Prediction — 24‑Hour Delivery Blueprint

This document is the execution blueprint for the *Machine Learning-Based Prediction of Air Pollutant Concentrations* assignment. It is written for rapid, low-risk delivery within **24 hours**, with a strong emphasis on: (1) getting the dataset into a clean, machine-learning-friendly “tidy” table, (2) generating leakage-safe lag/temporal features, and (3) training/evaluating **Linear Regression, Random Forest, XGBoost, and LSTM** using **RMSE, MAE, R²**, plus prediction-vs-actual plots (per the assignment PDF).

The dataset in this workspace is a collection of Excel workbooks under `Dataset/` with multiple monitoring locations (stations). A smart profiler (sampled) confirmed that the Excel layout is **non-standard** and requires a deterministic reshape step before any modeling.

---

## 1) Project Scope & Data Schema

### 1.1 Assignment scope (from the PDF)
- Objective: **Predict pollutant concentrations** using **pollutant + meteorological features**.
- Required pipeline steps:
  1. **Preprocessing** (clean + normalize)
  2. **Feature engineering** (lag + temporal features)
  3. Train models: **Linear Regression, Random Forest, XGBoost, LSTM**
  4. Evaluate: **RMSE, MAE, R²**
  5. Compare + visualize (**prediction vs actual**)

### 1.2 Workspace dataset structure (observed)
- Root folder: `Dataset/`
- Stations (top-level subfolders found):
  - `Bhatagaon DCR/`
  - `DCR AIIMS/`
  - `IGKV DCR/`
  - `SILTARA DCR/`
- File types: mixture of **`.xlsx` / `.xlsm` / `.xls`**.
- Workbooks are typically *month-level* files containing **many sheets**, commonly named like **`01`, `02`, `03`, …`31`** (day-of-month).
- Sampling indicates a **15-minute base** (labelled “15 Min” / “Quat / Quat. Hourly” in filenames and sheet metadata).

### 1.3 The critical schema reality: “row-embedded column names”
Across profiled workbooks, the “data” sheet is shaped like this:

- **Column headers** (row near the top): `Date:` plus many columns whose header text *looks like a timestamp* (e.g., `01-08-2022 00:15`, `02/01/2024 00:15`).  
- A few rows later, there is a row whose first cell contains a phrase like **`Date & Time`**.  
  - The *rest of that row* contains the **actual variable names** for each data column (examples observed):
    - Pollutants: `PM10`, `PM2.5`, `NO`, `NO2`, `NOX`, `SO2`, `O3`, `CO`, `NH3`, `BENZ` (and occasional others)
    - Meteorology: `TEMP`, `HUM`, `WS` (wind speed), `WD` (wind direction), `RG` (rain gauge), `SR` (solar radiation)
    - Many labels include station suffixes like `_SILTARA`, `__BHATAGAON`, `_IGKV`, `_AIIM(S)`; these must be normalized.
- Immediately after the variable-name row, a **units row** is commonly present (e.g., `ug/m3`, `m/sec`, `deg`, `degreC`).
- The actual measurement time series begins on the next rows where the first column becomes a parseable datetime string (e.g., `01-08-2022 00:15`, `01-08-2022 00:30`, …).

**Implication:** For each sheet, we must:
1. Detect the **variable-name row** (`Date & Time` row).
2. Use that row to **rename** the measurement columns.
3. Drop the units row (or keep it in metadata only).
4. Parse the first column as `timestamp`.

### 1.4 Canonical “tidy table” target schema (what every model will consume)
After ingestion + reshape, the canonical dataset will be a single table with:

**Index / keys**
- `timestamp` (timezone-naive; assume local time unless otherwise specified)
- `station` (one of: `BHATAGAON`, `AIIMS`, `IGKV`, `SILTARA` — derived from folder + variable suffix)

**Core numeric features** (as available)
- Pollutants: `PM2.5`, `PM10`, `NO`, `NO2`, `NOX`, `SO2`, `O3`, `CO`, `NH3`, `BENZ`, …
- Meteorology: `TEMP`, `HUM`, `WS`, `WD`, `RG`, `SR`, …

**Optional metadata columns**
- `source_file`, `source_sheet` (for traceability/debug)
- `freq_minutes` (expected 15)

### 1.5 Target variable(s)
To satisfy the assignment while staying deliverable in 24 hours:
- **Primary target (recommended):** `PM2.5`
- **Secondary targets (stretch / optional multi-target):** `PM10`, `NO2`, `SO2`, `O3`

Rationale:
- `PM2.5` and `PM10` are typically the most useful air-quality targets.
- The profiled sample indicates these variables appear broadly across stations.

---

## 2) Preprocessing Strategy

### 2.1 Ingestion + reshape (mandatory before preprocessing)
Per Excel sheet:
1. Read a limited region near the top to find:
   - `header_row` (contains `Date:` and many time-like column headers)
   - `var_row` where first column matches regex: `(?i)^\\s*date\\s*&\\s*time\\s*$`
   - `units_row = var_row + 1` (if it contains mostly strings like `ug/m3`, `m/sec`, `deg`)
2. Define:
   - `timestamp_col = first column` (usually labelled `Date:`)
   - `value_cols = all remaining columns`
3. Rename `value_cols` using the values found in `var_row` for those columns.
4. Keep rows after `units_row` where `timestamp_col` parses to datetime.
5. Coerce the renamed columns to numeric, treating non-numeric tokens as missing (see 2.2).
6. Append to an intermediate dataset store (Parquet recommended).

### 2.2 Standardize missing values and “status tokens”
Observed non-numeric tokens include values like:
- `Maint.`, `Maintenance`, `Calibration`, `Calib.`, `Power Off`, `Pw.Off`, `Link Fail`, `Warm Up`, `Z.Ref`, etc.

Rule set (deterministic):
- Convert any cell that matches (case-insensitive) one of:
  - `maint`, `calib`, `power`, `link`, `warm`, `ref`, `fail`, `off`
  into `NaN` in numeric columns.
- Convert empty strings, whitespace-only strings, and Excel blank cells to `NaN`.
- Convert “NaN-like” strings (`nan`, `na`, `null`, `none`) to `NaN`.

### 2.3 Datetime normalization (strict)
All timestamps must end up as `datetime64[ns]`:
- Accept both `DD-MM-YYYY HH:MM` and `DD/MM/YYYY HH:MM` (observed formats).
- Remove stray whitespace.
- If seconds are missing, assume `:00`.
- Enforce monotonic sorting per `station`.
- De-duplicate exact duplicate timestamps per station:
  - If duplicates occur, aggregate by **mean** for numeric columns (safe, simple).

### 2.4 Frequency standardization
Assume the base is **15-minute**:
- Validate that most consecutive `timestamp` differences are 15 minutes.
- If mixed frequencies exist:
  - Create a *single base* by resampling to 15-min grid per station.
  - Aggregation: `mean` for continuous values (pollutants/met), and keep `NaN` when all missing.

Optional fast-track simplification:
- Also generate an **hourly** dataset by resampling to 60 minutes with `mean`.
  - This reduces rows by ~4× and accelerates XGBoost/LSTM training.
  - Keep both versions; choose hourly for LSTM if time is tight.

### 2.5 Missing-value handling (leakage-safe)
For each station and each numeric column:
1. Compute missingness.
2. Apply a tiered fill strategy:
   - Short gaps: time interpolation (linear) **with a strict limit**:
     - 15-min data: fill gaps up to **4 steps** (≤ 1 hour)
     - hourly data: fill gaps up to **2 steps** (≤ 2 hours)
   - Remaining missing:
     - For **tree models** (RF/XGBoost): leave as `NaN` if model supports it (XGBoost does; RF does not).
     - For **Linear Regression / LSTM**: impute using training-set statistics:
       - `median` per station per feature (robust).

Important:
- Any imputation statistic must be computed **only on the training split** and then applied to val/test.

### 2.6 Outlier handling (minimal + safe)
Given the 24-hour constraint, avoid heavy outlier modeling:
- For each pollutant feature and target:
  - Clip to `[P0.5, P99.5]` computed on training data per station (winsorization).
- Optionally set negative values to `NaN` (many pollutant concentrations cannot be negative).

### 2.7 Normalization / scaling
Scaling is model-dependent:
- **Linear Regression**: `StandardScaler` on numeric features.
- **Random Forest**: no scaling required.
- **XGBoost**: no scaling required (but OK if applied consistently).
- **LSTM**: `StandardScaler` (or `MinMaxScaler`) fit on training data; apply to val/test.

---

## 3) Feature Engineering Logic

All features must be computed **per station** and must be **strictly causal** (use only times `< t` to predict `t`).

### 3.1 Core predictors (baseline feature set)
Use:
- Pollutant features (excluding the target itself if required for fairness comparisons; otherwise include it as an autoregressive predictor).
- Meteorological features: `TEMP`, `HUM`, `WS`, `WD`, `RG`, `SR` (if present).

### 3.2 Temporal (calendar) features (from timestamp)
For each row with timestamp `t`:
- `hour = t.hour`
- `day_of_week = t.dayofweek` (0=Mon … 6=Sun)
- `month = t.month`
- `day_of_year = t.dayofyear`
- `is_weekend = 1 if day_of_week in {5,6} else 0`

Recommended cyclical encoding (helps linear + LSTM):
- `hour_sin = sin(2π * hour / 24)`
- `hour_cos = cos(2π * hour / 24)`
- `dow_sin = sin(2π * day_of_week / 7)`
- `dow_cos = cos(2π * day_of_week / 7)`

### 3.3 Lag features (t−n)
Let the base sampling interval be **Δ = 15 minutes**.

Define lag steps (fast-track, high value):
- `L = {1, 2, 3, 4, 8, 12, 24, 48, 96}`

Meaning:
- `1` = 15 min
- `4` = 1 hour
- `12` = 3 hours
- `24` = 6 hours
- `96` = 24 hours

For each selected predictor variable `x` (including the target `y` for autoregression):
- `x_lag_k(t) = x(t - k*Δ)` for each `k ∈ L`

This is implemented as:
- `x_lag_k = x.shift(k)` on the station-sorted time series.

### 3.4 Rolling-window statistics (trend/smoothing)
Use rolling windows in **steps**, aligned with 15-min data:
- `W = {4, 12, 24, 48, 96}`  → {1h, 3h, 6h, 12h, 24h}

For each predictor `x`:
- `x_rollmean_w(t) = mean( x(t-Δ), x(t-2Δ), …, x(t-w*Δ) )`
- `x_rollstd_w(t) = std( ... )`

Implementation notes (leakage-safe):
- Always compute on shifted series: `x.shift(1).rolling(window=w).mean()`

### 3.5 Wind direction handling (WD)
`WD` is circular (0–360 degrees). Use vectorization:
- `wd_rad = WD * π / 180`
- `wd_sin = sin(wd_rad)`
- `wd_cos = cos(wd_rad)`

Then drop raw `WD` (or keep both).

### 3.6 Feature/target alignment and dropped rows
Because lag/rolling features create leading `NaN` rows:
- After features are created, drop rows with any missing in:
  - the target `y(t)`
  - required lag features (at least for LR/LSTM)
- For tree models, you may keep some missing if supported; however, for speed and consistency, the fast-track plan is to **drop** rows missing the key lag set.

---

## 4) Model Architecture (Fast-Track)

The goal is to produce comparable models quickly and reliably.

### 4.1 Common training protocol (all models)
- Split strategy (time-series split, per station):
  - Train: earliest 70%
  - Validation: next 15%
  - Test: last 15%
  - (Adjust if data is sparse; never shuffle.)
- Metrics:
  - RMSE = √mean((ŷ − y)²)
  - MAE = mean(|ŷ − y|)
  - R²
- Baselines (must include):
  - Persistence baseline: `ŷ(t) = y(t-1)` (or `y(t-4)` for 1-hour persistence)

### 4.2 Linear Regression (regularized; fast and stable)
Library: `scikit-learn`
- Pipeline:
  - `StandardScaler`
  - `Ridge` regression (safer than plain OLS under collinearity)
- Suggested parameters:
  - `alpha ∈ {0.1, 1.0, 10.0}` (choose best on validation)
  - `fit_intercept=True`

### 4.3 Random Forest Regressor (robust nonlinear baseline)
Library: `scikit-learn`
- Suggested fast parameters:
  - `n_estimators=300`
  - `max_depth=20` (or `None` if dataset small after filtering)
  - `min_samples_leaf=2`
  - `max_features="sqrt"`
  - `n_jobs=-1`
  - `random_state=42`

Note: RF does not accept `NaN` → ensure imputation or row-drop before training.

### 4.4 XGBoost Regressor (best accuracy per unit time)
Library: `xgboost`
- Suggested fast parameters (CPU, small-to-mid dataset):
  - `n_estimators=1500` with early stopping
  - `learning_rate=0.03`
  - `max_depth=6`
  - `subsample=0.8`
  - `colsample_bytree=0.8`
  - `reg_lambda=1.0`
  - `objective="reg:squarederror"`
  - `tree_method="hist"`
  - `eval_metric="rmse"`
- Use validation set for `early_stopping_rounds=50`.

XGBoost handles missing values natively; still prefer consistent preprocessing.

### 4.5 LSTM (lightweight, fast-compiling)
Goal: satisfy the assignment’s deep-learning requirement with minimal engineering risk.

Recommended approach:
- Use a **single-layer LSTM** with a small hidden size.
- Train on **hourly** data if 15-min data is too large (4× fewer steps).

Input construction:
- Sequence length `T`:
  - 15-min data: `T = 96` (past 24 hours)
  - hourly data: `T = 24` (past 24 hours)
- Features per timestep: scaled numeric predictors (pollutants + met + time encodings).

Architecture (Keras `tf.keras`, minimal):
- `Input(shape=(T, n_features))`
- `LSTM(32, dropout=0.1, recurrent_dropout=0.0, return_sequences=False)`
- `Dense(16, activation="relu")`
- `Dense(1)`  (single-target; wrap for multi-target later if needed)

Training hyperparameters (deadline-friendly):
- `optimizer=Adam(learning_rate=1e-3)`
- `loss="mse"`
- `batch_size=128`
- `epochs=20` with early stopping:
  - `EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)`

Critical guardrails:
- No data leakage: sequences for time `t` must only include values `< t`.
- Scale using training data only; apply same scaler to val/test.

---

## 5) Execution Phases (Script Checklist)

This is the minimal set of scripts to produce a complete, reportable submission.

### Phase A — Data ingestion (reshape Excel → tidy table)
1. `1_ingest_excel.py`
   - Walk `Dataset/` recursively.
   - For each workbook:
     - For each sheet:
       - Detect `Date & Time` row
       - Rename columns using that row
       - Parse timestamps
       - Coerce numeric columns; convert status tokens → NaN
       - Append rows to a unified dataset
   - Output:
     - `artifacts/data_15min.parquet` (canonical)
     - Optional: `artifacts/data_hourly.parquet` (resampled)

### Phase B — Preprocessing (cleaning + splits + scalers)
2. `2_preprocess.py`
   - Load `data_15min.parquet` (or hourly).
   - Apply:
     - de-dup timestamps per station
     - missing-value rules + limited interpolation
     - winsorization (train-only percentiles)
   - Create time-based splits:
     - `train.parquet`, `val.parquet`, `test.parquet`
   - Persist preprocessing artifacts:
     - `artifacts/imputer.pkl` (if used)
     - `artifacts/scaler.pkl` (LR/LSTM)

### Phase C — Feature engineering (lag/rolling/time encodings)
3. `3_features.py`
   - Load split parquet.
   - Create temporal features.
   - Create lag and rolling features per station using the exact rules in Section 3.
   - Output:
     - `artifacts/features_train.parquet`
     - `artifacts/features_val.parquet`
     - `artifacts/features_test.parquet`

### Phase D — Model training (3 classical models)
4. `4_train_sklearn.py`
   - Train Linear (Ridge) + RandomForest on `features_train`.
   - Tune only the smallest knob set (Ridge alpha, RF depth) using validation.
   - Save:
     - `artifacts/model_ridge.pkl`
     - `artifacts/model_rf.pkl`

5. `5_train_xgb.py`
   - Train XGBoost with early stopping on val.
   - Save:
     - `artifacts/model_xgb.json` (or `.pkl`)

### Phase E — LSTM training (sequence model)
6. `6_train_lstm.py`
   - Build sequences (T=96 for 15-min or T=24 for hourly).
   - Train lightweight LSTM (Section 4.5).
   - Save:
     - `artifacts/model_lstm.keras`

### Phase F — Evaluation + plots + final report
7. `7_evaluate.py`
   - Evaluate all models + persistence baseline on test set.
   - Compute RMSE/MAE/R² (overall + per station).
   - Output tables:
     - `artifacts/metrics.csv`
     - `artifacts/metrics_by_station.csv`

8. `8_plots.py`
   - Generate:
     - Prediction vs actual line plots (time-series)
     - Scatter plots (ŷ vs y)
   - Save to:
     - `artifacts/plots/`

9. `REPORT.md` (final deliverable)
   - Dataset description (schema + preprocessing decisions)
   - Feature engineering details (exact lags/windows)
   - Model configs
   - Metrics table + plots
   - Comparative analysis + concluding recommendation

