from __future__ import annotations

import argparse
import ctypes
import json
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# NOTE:
# This script is intentionally written as a "single entrypoint" module, but with
# strict functional modularity. Each function has explicit type hints and a
# Google-style docstring, and the pipeline is arranged to reduce leakage risk.

try:
    import numpy as np
    import pandas as pd
    np.seterr(all="ignore")
    # pandas 3.0 uses StringDtype by default — disable to keep numpy dtype
    # checks working as expected with pandas 2.x behaviour.
    try:
        pd.options.future.infer_string = False
    except AttributeError:
        pass
except Exception as e:  # pragma: no cover
    print(
        "Missing runtime dependencies for `2_preprocess_and_features.py`.\n"
        "This script requires `numpy` + `pandas` (and `pyarrow` for Parquet I/O).\n"
        f"Import error: {e}\n",
        file=sys.stderr,
    )
    raise SystemExit(10)


TARGET_COL: str = "PM2.5"
FREQ: str = "15min"

LAGS: List[int] = [1, 2, 3, 4, 8, 12, 24, 48, 96]
ROLL_WINDOWS: List[int] = [4, 12, 24, 48, 96]

# We need enough historical context so that, for validation/test, lag/rolling
# features for the first row in the segment can be computed using prior data
# (from the preceding split) without recomputing the entire history.
CONTEXT_STEPS: int = max(max(LAGS), max(ROLL_WINDOWS)) + 1

# These columns are produced in Phase A. We keep them for traceability, but we
# do not use them as features.
META_COLS: Tuple[str, ...] = ("station", "timestamp", "source_file", "source_sheet")


def setup_logging(level: str) -> None:
    """Configure global logging format and level.

    Args:
        level: Logging level string (e.g., "INFO", "DEBUG").
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def system_memory_percent() -> float:
    """Return approximate system-wide memory usage percent.

    This function first tries `psutil` (if installed). If unavailable, it falls back
    to Windows `GlobalMemoryStatusEx` via ctypes.

    Returns:
        Memory usage percent in [0, 100].
    """
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().percent)
    except Exception:
        pass

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    if not ok:
        return 0.0
    return float(stat.dwMemoryLoad)


def stop_if_memory_high(threshold_percent: float) -> None:
    """Stop execution if system memory usage exceeds a threshold.

    This is a safety rail to prevent thrashing / crashes when concatenating very
    wide or long time-series data.

    Args:
        threshold_percent: Stop if memory usage >= this value.
    """
    mem = system_memory_percent()
    if mem >= threshold_percent:
        raise SystemExit(f"Stopping: system memory usage {mem:.1f}% >= {threshold_percent:.1f}%")


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame.

    Args:
        path: Path to the Parquet file.

    Returns:
        Loaded DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet: {path}")
    # Parquet I/O requires `pyarrow` or `fastparquet`. We keep this explicit so
    # failures are actionable.
    return pd.read_parquet(path)


def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Ensure a column is parsed as datetime64[ns].

    Args:
        df: Input DataFrame.
        col: Column name to parse.

    Returns:
        DataFrame with parsed datetime column.
    """
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def canonicalize_variable_name(name: str, station: str) -> str:
    """Canonicalize a variable name to a stable base form.

    Phase A intentionally used the "exact string" found in the Excel variable-name row.
    Those labels often include station suffixes (e.g., `PM2.5__BHATAGAON`, `NO2_SILTARA`).
    For modeling and especially for selecting the primary target, we need a consistent
    name across stations.

    Strategy:
    - Normalize whitespace and casing
    - Normalize PM2.5/PM10 formats
    - Drop station suffix patterns: `__STATION`, `_STATION`, and some known variants
    - Keep the remaining base name

    Args:
        name: Original column name.
        station: Station identifier (e.g., "BHATAGAON").

    Returns:
        Canonicalized variable name.
    """
    s = str(name).strip()
    s = re.sub(r"\s+", " ", s)
    s_up = s.upper()

    # Normalize frequent pollutant label variations.
    s_up = s_up.replace("PM 2.5", "PM2.5").replace("PM2_5", "PM2.5")
    s_up = s_up.replace("PM 10", "PM10")

    # Handle double-underscore suffixes: `NO2__BHATAGAON` -> `NO2`
    if "__" in s_up:
        s_up = s_up.split("__", 1)[0].strip()

    # Handle single underscore suffix if it matches the station name or common
    # partials (AIIM/AIIMS).
    if "_" in s_up:
        base, suffix = s_up.rsplit("_", 1)
        suffix = suffix.strip()
        station_up = station.upper()
        if suffix in {station_up, station_up.replace(" ", ""), "AIIM", "AIIMS"}:
            s_up = base.strip()

    # Some files embed station name without underscore (e.g., `SO2SILTARA`).
    station_alpha = re.sub(r"[^A-Z]", "", station.upper())
    if station_alpha and s_up.endswith(station_alpha) and len(s_up) > len(station_alpha) + 1:
        s_up = s_up[: -len(station_alpha)].strip("_ ").strip()

    return s_up


@dataclass(frozen=True)
class ColumnMapping:
    """Represents a resolved mapping for station-specific columns."""

    station: str
    chosen: Dict[str, str]  # canonical_name -> original_column


def resolve_station_column_mapping(df: pd.DataFrame, station: str) -> ColumnMapping:
    """Resolve a per-station mapping from canonical variable names to original columns.

    Why this exists:
    - Phase A output may contain multiple station-suffixed variations for the same
      base variable name.
    - We want *one* column per canonical variable per station (choose the densest).

    Leakage note:
    - This mapping is based only on column missingness counts, not on label values.
      It does not use future information in a way that affects model evaluation.

    Args:
        df: Station-filtered DataFrame.
        station: Station name.

    Returns:
        ColumnMapping defining which original column to keep for each canonical name.
    """
    candidates = [c for c in df.columns if c not in META_COLS]
    if not candidates:
        return ColumnMapping(station=station, chosen={})

    by_canon: Dict[str, List[str]] = {}
    for c in candidates:
        canon = canonicalize_variable_name(c, station=station)
        by_canon.setdefault(canon, []).append(c)

    chosen: Dict[str, str] = {}
    for canon, cols in by_canon.items():
        if len(cols) == 1:
            chosen[canon] = cols[0]
            continue
        # Choose the column with the most non-null numeric values.
        best_col = None
        best_score = -1
        for c in cols:
            series = pd.to_numeric(df[c], errors="coerce")
            score = int(series.notna().sum())
            if score > best_score:
                best_col = c
                best_score = score
        chosen[canon] = best_col if best_col is not None else cols[0]

    return ColumnMapping(station=station, chosen=chosen)


def standardize_columns_per_station(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize measurement columns so canonical names are consistent across stations.

    Output columns:
    - Meta columns: `station`, `timestamp` (+ optional `source_file`, `source_sheet`)
    - Canonical measurement columns: e.g., `PM2.5`, `PM10`, `NO2`, `TEMP`, ...

    Important:
    - This drops duplicate synonyms within a station (keeping the densest).
    - This step is essential so the target variable `PM2.5` can be referenced
      consistently across all stations.

    Args:
        df: Raw Phase A DataFrame.

    Returns:
        Standardized DataFrame.
    """
    if "station" not in df.columns or "timestamp" not in df.columns:
        raise ValueError("Expected columns `station` and `timestamp` in Phase A output.")

    out_frames: List[pd.DataFrame] = []
    for station, sdf in df.groupby("station", sort=False):
        mapping = resolve_station_column_mapping(sdf, station=str(station))
        keep_cols: List[str] = ["station", "timestamp"]
        for c in ("source_file", "source_sheet"):
            if c in sdf.columns:
                keep_cols.append(c)

        # Create canonical measurement columns
        canon_cols: Dict[str, pd.Series] = {}
        for canon, orig in mapping.chosen.items():
            canon_cols[canon] = sdf[orig]

        station_out = sdf[keep_cols].copy()
        for canon, series in canon_cols.items():
            station_out[canon] = series

        out_frames.append(station_out)

    out = pd.concat(out_frames, ignore_index=True, sort=False)
    return out


def align_to_15min_grid(df_station: pd.DataFrame) -> pd.DataFrame:
    """Force a strict 15-minute time grid for one station.

    Steps:
    1) Sort by timestamp.
    2) Aggregate exact duplicate timestamps by mean for numeric columns.
    3) Reindex onto a complete 15-min date_range from min..max.

    Leakage note:
    - Grid alignment is a structural normalization and does not incorporate future
      target values. It just creates explicit missing timestamps as NaNs.

    Args:
        df_station: DataFrame for a single station.

    Returns:
        Grid-aligned station DataFrame.
    """
    if df_station.empty:
        return df_station

    df_station = df_station.sort_values("timestamp").copy()

    # Aggregate duplicates via mean for numeric columns.
    numeric_cols = [c for c in df_station.columns if c not in ("station", "timestamp", "source_file", "source_sheet")]
    # Coerce numeric cols to numeric before mean to avoid accidental string propagation.
    for c in numeric_cols:
        df_station[c] = pd.to_numeric(df_station[c], errors="coerce")

    if df_station.duplicated(subset=["timestamp"]).any():
        df_station = (
            df_station.groupby(["station", "timestamp"], as_index=False)[numeric_cols]
            .mean(numeric_only=True)
            .sort_values("timestamp")
        )

    station = str(df_station["station"].iloc[0])
    start = df_station["timestamp"].min()
    end = df_station["timestamp"].max()

    # Strict 15-minute grid. Missing timestamps become NaN.
    full_index = pd.date_range(start=start, end=end, freq=FREQ)
    df_station = df_station.set_index("timestamp").reindex(full_index)
    df_station.index.name = "timestamp"
    df_station = df_station.reset_index()
    df_station["station"] = station

    return df_station.sort_values("timestamp").reset_index(drop=True)


def split_station_time_series(df_station: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a station time series into train/val/test by chronological order.

    Split ratios:
    - Train: earliest 70%
    - Val: next 15%
    - Test: last 15%

    Args:
        df_station: Station DataFrame in chronological order.

    Returns:
        (train_df, val_df, test_df)
    """
    n = int(df_station.shape[0])
    if n == 0:
        return df_station.copy(), df_station.copy(), df_station.copy()

    train_end = int(math.floor(0.70 * n))
    val_end = int(math.floor(0.85 * n))

    train = df_station.iloc[:train_end].copy()
    val = df_station.iloc[train_end:val_end].copy()
    test = df_station.iloc[val_end:].copy()
    return train, val, test


def handle_missing_values(df: pd.DataFrame, max_forward_steps: int) -> pd.DataFrame:
    """Impute short gaps using linear interpolation limited to 1 hour (4 x 15-min steps).

    Why interpolation is constrained:
    - Small gaps often come from telemetry dropouts or short maintenance periods.
    - Long-gap interpolation fabricates extended segments and can mislead models.

    Leakage control:
    - Interpolation is applied *within each split segment* (train/val/test), so
      information does not leak across split boundaries.
    - Note: Linear interpolation within a segment uses points on both sides of a
      gap *inside that segment*. This is accepted here as a data-cleaning step
      constrained to <= 1 hour, but it should be revisited for strict real-time
      forecasting settings.

    Args:
        df: Input DataFrame containing `timestamp` and numeric columns.
        max_forward_steps: Maximum number of consecutive NaNs to fill forward.

    Returns:
        DataFrame with interpolated numeric columns.
    """
    if df.empty:
        return df

    out = df.copy()
    out = out.sort_values("timestamp").reset_index(drop=True)
    out = out.set_index("timestamp")

    numeric_cols = [
        c for c in out.columns if c not in ("station", "source_file", "source_sheet") and pd.api.types.is_numeric_dtype(out[c])
    ]

    # Ensure numeric dtype; interpolation requires float-like.
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # `method="time"` uses the datetime index spacing, which is exactly what we want
    # after grid alignment. `limit` ensures we fill only short gaps.
    out[numeric_cols] = out[numeric_cols].interpolate(
        method="time",
        limit=max_forward_steps,
        limit_area="inside",  # Only interpolate gaps with known endpoints (avoid extrapolation).
    )

    out = out.reset_index()
    return out


def clean_measurement_data(df: pd.DataFrame) -> pd.DataFrame:
    """Set negative values and extreme placeholders to NaN for measurement columns.

    This prevents sentinel values (e.g., -2B) from corrupting metrics and model training.
    """
    out = df.copy()
    meas_cols = [c for c in out.columns if c not in META_COLS]
    for c in meas_cols:
        if not pd.api.types.is_numeric_dtype(out[c]):
            series = pd.to_numeric(out[c], errors="coerce")
        else:
            series = out[c]

        # Pollutants and meteorological measurements cannot be negative.
        # This catches -2e9 placeholders as well as regular negative noise.
        out[c] = series.where(series >= 0, np.nan)

    return out


def cap_target_by_train_quantile(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    quantile: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[float]]:
    """Cap extreme target values using a quantile computed on the TRAIN split only.

    Why:
    - Some stations contain extreme PM2.5 spikes (or remaining placeholders) that
      can dominate squared-error metrics and destabilize tree models.
    - Capping is applied using a threshold learned from the training split only
      (no leakage).

    Args:
        train_df: Training split for one station (cleaned + imputed).
        val_df: Validation split for one station (cleaned + imputed).
        test_df: Test split for one station (cleaned + imputed).
        target_col: Target column name (e.g., "PM2.5").
        quantile: Quantile in (0, 1]. Typical: 0.995 or 0.99. If <=0, no-op.

    Returns:
        (train_capped, val_capped, test_capped, cap_value)
    """
    if quantile <= 0.0 or quantile > 1.0:
        return train_df, val_df, test_df, None
    if target_col not in train_df.columns:
        return train_df, val_df, test_df, None

    s = pd.to_numeric(train_df[target_col], errors="coerce")
    if not s.notna().any():
        return train_df, val_df, test_df, None
    cap_value = float(s.quantile(quantile))
    if not np.isfinite(cap_value) or cap_value <= 0.0:
        return train_df, val_df, test_df, None

    def _cap(df_in: pd.DataFrame) -> pd.DataFrame:
        out = df_in.copy()
        out[target_col] = pd.to_numeric(out[target_col], errors="coerce").clip(lower=0.0, upper=cap_value)
        return out

    return _cap(train_df), _cap(val_df), _cap(test_df), cap_value


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical temporal features and a weekend flag.

    Features:
    - hour_sin, hour_cos (24h cycle)
    - dow_sin, dow_cos (7-day cycle)
    - is_weekend (Sat/Sun)

    Args:
        df: Input DataFrame with `timestamp`.

    Returns:
        DataFrame with additional temporal feature columns.
    """
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], errors="coerce")
    hour = ts.dt.hour.astype("float64")
    dow = ts.dt.dayofweek.astype("float64")

    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype("int8")
    return out


def add_wind_direction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wind direction (WD degrees) into circular features.

    `WD` is circular (0..360). Using raw degrees creates artificial discontinuities
    at 0/360. Representing direction as sin/cos preserves circular continuity.

    Args:
        df: Input DataFrame possibly containing a `WD` column.

    Returns:
        DataFrame with `wd_sin`, `wd_cos` added and raw `WD` dropped (if present).
    """
    if "WD" not in df.columns:
        return df

    out = df.copy()
    wd = pd.to_numeric(out["WD"], errors="coerce")
    wd_rad = wd * (np.pi / 180.0)
    out["wd_sin"] = np.sin(wd_rad)
    out["wd_cos"] = np.cos(wd_rad)

    # Drop raw WD to avoid collinearity and discontinuity artifacts.
    out = out.drop(columns=["WD"])
    return out


def infer_base_feature_columns(df: pd.DataFrame, target_col: str, require_any_nonnull: bool = True) -> List[str]:
    """Infer which columns should receive lag/rolling features.

    Per spec, lag/rolling should apply to pollutant + meteorological variables,
    including the target `PM2.5`. In Phase A, measurement columns are numeric-ish
    and everything else is metadata or engineered.

    We take all numeric columns except temporal engineered columns and exclude
    identifiers.

    Args:
        df: DataFrame after basic cleaning and optional WD conversion.
        target_col: Name of target column (kept included).

    Returns:
        List of base feature columns to lag/roll.
    """
    excluded = {
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "station",
        "timestamp",
        "source_file",
        "source_sheet",
    }
    base: List[str] = []
    for c in df.columns:
        if c in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            # IMPORTANT:
            # - If `require_any_nonnull=True` (default), we include only variables that
            #   have at least one observed value in the provided frame. This keeps the
            #   feature space smaller and avoids generating huge numbers of all-NaN
            #   lag/rolling columns.
            # - For schema stability across train/val/test, callers may compute the
            #   base column list from a wider context (e.g., the train segment) and
            #   pass it into `build_features_with_context()`. In that case we do NOT
            #   want per-segment missingness to change which engineered columns exist.
            if (not require_any_nonnull) or (not df[c].isna().all()):
                base.append(c)
    # Ensure target is included (if numeric)
    if target_col in df.columns and target_col not in base and pd.api.types.is_numeric_dtype(df[target_col]):
        base.append(target_col)
    return base


def add_lag_features(
    df: pd.DataFrame,
    group_col: str,
    cols: Sequence[str],
    lags: Sequence[int],
) -> Tuple[pd.DataFrame, List[str]]:
    """Generate lag features per station using groupby().shift().

    Leakage control:
    - Lags are strictly causal: `x_lag_k(t)` uses `x(t-k)` only.
    - Computation is done within each station group to prevent cross-station mixing.
    """
    new_cols_dict: Dict[str, pd.Series] = {}

    # Accumulate all lags in a dictionary first to avoid DataFrame fragmentation.
    for c in cols:
        for k in lags:
            new_col = f"{c}_lag_{k}"
            new_cols_dict[new_col] = df.groupby(group_col, sort=False)[c].shift(k)

    # Use pd.concat(axis=1) for a single structural update.
    if new_cols_dict:
        out = pd.concat([df, pd.DataFrame(new_cols_dict, index=df.index)], axis=1)
    else:
        out = df.copy()

    return out, list(new_cols_dict.keys())


def add_rolling_mean_features(
    df: pd.DataFrame,
    group_col: str,
    cols: Sequence[str],
    windows: Sequence[int],
) -> pd.DataFrame:
    """Generate leakage-safe rolling mean features per station."""
    new_cols_dict: Dict[str, pd.Series] = {}

    for c in cols:
        shifted = df.groupby(group_col, sort=False)[c].shift(1)
        for w in windows:
            # The rolling mean is computed within each station group, on the
            # *shifted* series, so the statistic at time `t` only uses values
            # strictly before `t` (leakage-safe).
            rolled = (
                shifted.groupby(df[group_col], sort=False)
                .rolling(window=w)
                .mean()
                .reset_index(level=0, drop=True)
            )
            new_cols_dict[f"{c}_rollmean_{w}"] = rolled

    if new_cols_dict:
        out = pd.concat([df, pd.DataFrame(new_cols_dict, index=df.index)], axis=1)
    else:
        out = df.copy()
    return out


def drop_rows_with_required_nonnull(
    df: pd.DataFrame,
    target_col: str,
    required_cols: Sequence[str],
) -> pd.DataFrame:
    """Drop rows missing the target or required engineered features.

    Rationale:
    - Many learners (Linear Regression, RF) cannot handle NaN.
    - Even for models that can, missing lags weaken comparability and complicate
      reporting under a 24-hour deadline.

    Args:
        df: Input DataFrame.
        target_col: Target column name.
        required_cols: Columns that must be non-null (e.g., newly created lags).

    Returns:
        Filtered DataFrame.
    """
    cols = [target_col] + list(required_cols)
    keep = df.dropna(subset=cols)
    return keep.reset_index(drop=True)


@dataclass(frozen=True)
class StandardScalerParams:
    """Parameters for a simple per-column standardization."""

    mean: Dict[str, float]
    std: Dict[str, float]


def fit_standard_scaler(df_train: pd.DataFrame, feature_cols: Sequence[str]) -> StandardScalerParams:
    """Fit a simple standard scaler on training features only.

    Leakage control:
    - We compute mean/std *only* from the training split and apply to val/test.

    Args:
        df_train: Training DataFrame.
        feature_cols: Feature columns to scale.

    Returns:
        StandardScalerParams with mean/std per feature.
    """
    mean: Dict[str, float] = {}
    std: Dict[str, float] = {}
    for c in feature_cols:
        series = pd.to_numeric(df_train[c], errors="coerce")
        mu = float(series.mean(skipna=True))
        sigma = float(series.std(skipna=True))
        if not np.isfinite(sigma) or sigma == 0.0:
            sigma = 1.0
        mean[c] = mu
        std[c] = sigma
    return StandardScalerParams(mean=mean, std=std)


def apply_standard_scaler(df: pd.DataFrame, params: StandardScalerParams, feature_cols: Sequence[str]) -> pd.DataFrame:
    """Apply standard scaling to feature columns using pre-fit parameters.

    Args:
        df: Input DataFrame.
        params: Standard scaler parameters.
        feature_cols: Columns to scale.

    Returns:
        DataFrame with scaled feature columns.
    """
    out = df.copy()
    for c in feature_cols:
        if c not in out.columns:
            continue
        mu = params.mean.get(c, 0.0)
        sigma = params.std.get(c, 1.0)
        out[c] = (pd.to_numeric(out[c], errors="coerce") - mu) / sigma
    return out


def save_scaler_params(path: Path, params: StandardScalerParams) -> None:
    """Persist scaler params to JSON for reproducibility.

    Args:
        path: Output JSON path.
        params: Scaler parameters.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"mean": params.mean, "std": params.std}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame as parquet.

    Args:
        df: DataFrame to write.
        path: Destination parquet path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def build_features_with_context(
    df_history: pd.DataFrame,
    df_current: pd.DataFrame,
    target_col: str,
    memory_threshold: float,
    base_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Compute features for a split using limited historical context.

    We attach a short tail of *previous split data* so that lag/rolling features
    for the first rows in the current split can be computed causally.

    Args:
        df_history: Tail rows from previous split (already cleaned/imputed).
        df_current: Current split rows (already cleaned/imputed).
        target_col: Primary target column.
        memory_threshold: System memory safety threshold.

    Returns:
        Feature-engineered DataFrame for the current split (history rows removed).
    """
    stop_if_memory_high(memory_threshold)

    # Combine and mark boundaries so we can drop the history rows afterwards.
    hist = df_history.copy()
    cur = df_current.copy()
    hist["__segment"] = "history"
    cur["__segment"] = "current"
    combined = pd.concat([hist, cur], ignore_index=True, sort=False)

    # Wind direction conversion first so lags can apply to wd_sin/cos (instead of raw WD).
    combined = add_wind_direction_features(combined)

    # Temporal features do not depend on future values.
    combined = add_temporal_features(combined)

    # Determine base features (pollutants + meteorology) to lag/roll.
    #
    # Why `base_cols` is a parameter:
    # - If we infer base columns *per segment*, a variable that is all-NaN within
    #   validation/test would be excluded and its engineered columns would simply
    #   not exist for that segment.
    # - That causes schema drift between train/val/test (which previously broke
    #   RF/XGB training and invalidated early stopping).
    if base_cols is None:
        base_cols = infer_base_feature_columns(combined, target_col=target_col, require_any_nonnull=True)
    else:
        base_cols = list(base_cols)

    # Ensure all requested base columns exist (create all-NaN placeholders if not).
    for c in base_cols:
        if c not in combined.columns:
            combined[c] = np.nan

    # Create lag features and record which ones were created (for later NaN dropping).
    combined, lag_cols = add_lag_features(combined, group_col="station", cols=base_cols, lags=LAGS)

    # Create rolling mean features using shifted series to prevent leakage.
    combined = add_rolling_mean_features(combined, group_col="station", cols=base_cols, windows=ROLL_WINDOWS)

    # Keep only the rows in the current segment.
    combined = combined[combined["__segment"] == "current"].drop(columns=["__segment"])

    # Relaxed split sparsity fix: only drop rows where the target or its lags are missing.
    # This ensures that a station isn't dropped entirely just because a secondary
    # meteorological variable is missing.
    target_lags = [c for c in lag_cols if c.startswith(f"{target_col}_lag_")]
    combined = drop_rows_with_required_nonnull(combined, target_col=target_col, required_cols=target_lags)

    stop_if_memory_high(memory_threshold)
    return combined


def main() -> int:
    """CLI entrypoint for preprocessing + feature engineering."""
    parser = argparse.ArgumentParser(description="Preprocess Phase A parquet and generate leakage-safe features.")
    parser.add_argument("--in-parquet", type=str, default=str(Path("artifacts") / "data_15min.parquet"))
    parser.add_argument("--out-train", type=str, default=str(Path("artifacts") / "features_train.parquet"))
    parser.add_argument("--out-val", type=str, default=str(Path("artifacts") / "features_val.parquet"))
    parser.add_argument("--out-test", type=str, default=str(Path("artifacts") / "features_test.parquet"))
    parser.add_argument("--scaler-json", type=str, default=str(Path("artifacts") / "standard_scaler.json"))
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--memory-threshold", type=float, default=80.0)
    parser.add_argument("--target", type=str, default=TARGET_COL)
    parser.add_argument(
        "--schema-stable",
        action="store_true",
        help="Force train/val/test to share the exact same columns (based on train).",
    )
    parser.add_argument(
        "--schema-json",
        type=str,
        default=str(Path("artifacts") / "feature_schema.json"),
        help="Where to save the final feature column schema when --schema-stable is enabled.",
    )
    parser.add_argument(
        "--target-cap-quantile",
        type=float,
        default=0.0,
        help="If >0, cap extreme target values per station using this TRAIN quantile (e.g., 0.995).",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    in_path = Path(args.in_parquet).resolve()
    out_train = Path(args.out_train).resolve()
    out_val = Path(args.out_val).resolve()
    out_test = Path(args.out_test).resolve()
    scaler_path = Path(args.scaler_json).resolve()
    target_col = str(args.target)

    logging.info("Loading Phase A parquet: %s", in_path)
    df_raw = load_parquet(in_path)
    stop_if_memory_high(float(args.memory_threshold))

    # Ensure timestamp dtype is datetime64 and drop unparsable rows.
    df_raw = ensure_datetime(df_raw, col="timestamp")
    df_raw = df_raw.dropna(subset=["station", "timestamp"]).copy()

    # Standardize columns per station to canonical variable names (critical for `PM2.5`).
    logging.info("Standardizing measurement column names per station...")
    df = standardize_columns_per_station(df_raw)
    df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)

    if target_col not in df.columns:
        # We fail fast here because downstream operations depend on a stable target.
        raise SystemExit(
            f"Target column `{target_col}` not found after canonicalization. "
            "Inspect Phase A output for how PM2.5 is labeled."
        )

    # Coerce measurement columns to numeric early so grid aggregation is consistent.
    meas_cols = [c for c in df.columns if c not in META_COLS]
    for c in meas_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Data integrity fix: remove negative/sentinel values before splitting/features.
    logging.info("Cleaning sentinel/negative values from measurement data...")
    df = clean_measurement_data(df)

    # Process per station to enforce strict grid and leakage-aware splitting.
    train_parts: List[pd.DataFrame] = []
    val_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []
    per_station_base_cols: Dict[str, List[str]] = {}

    for station, sdf in df.groupby("station", sort=False):
        logging.info("Station=%s | rows=%d", station, sdf.shape[0])
        sdf = sdf[["station", "timestamp"] + [c for c in sdf.columns if c not in ("station", "timestamp")]].copy()
        sdf = sdf.sort_values("timestamp").reset_index(drop=True)

        # 1) Grid alignment & duplicate aggregation per station.
        aligned = align_to_15min_grid(sdf)

        # 7) Time-series split (strict chronological; no shuffle).
        train_raw, val_raw, test_raw = split_station_time_series(aligned)

        # 2) Imputation (within-split) to avoid cross-split leakage.
        train_clean = handle_missing_values(train_raw, max_forward_steps=4)
        val_clean = handle_missing_values(val_raw, max_forward_steps=4)
        test_clean = handle_missing_values(test_raw, max_forward_steps=4)

        # Optional target capping (leakage-safe): compute cap from TRAIN only, then apply to all.
        cap_q = float(getattr(args, "target_cap_quantile", 0.0))
        if cap_q > 0.0:
            train_clean, val_clean, test_clean, cap_value = cap_target_by_train_quantile(
                train_df=train_clean,
                val_df=val_clean,
                test_df=test_clean,
                target_col=target_col,
                quantile=cap_q,
            )
            if cap_value is not None:
                logging.info("Station=%s | target cap @q=%.3f => %.3f", station, cap_q, cap_value)

        # Schema stability:
        # Infer base columns once per station from the TRAIN segment so that
        # engineered columns exist consistently across train/val/test even if a
        # sensor goes fully missing in a later segment.
        template = add_temporal_features(add_wind_direction_features(train_clean.copy()))
        base_cols_station = infer_base_feature_columns(template, target_col=target_col, require_any_nonnull=True)
        per_station_base_cols[str(station)] = list(base_cols_station)

        # Feature generation with history context:
        # - Train: no history
        train_feat = build_features_with_context(
            df_history=train_clean.iloc[:0].copy(),
            df_current=train_clean,
            target_col=target_col,
            memory_threshold=float(args.memory_threshold),
            base_cols=base_cols_station,
        )

        # - Val: provide tail of train
        val_hist = train_clean.tail(CONTEXT_STEPS)
        val_feat = build_features_with_context(
            df_history=val_hist,
            df_current=val_clean,
            target_col=target_col,
            memory_threshold=float(args.memory_threshold),
            base_cols=base_cols_station,
        )

        # - Test: provide tail of (train + val)
        tv_hist = pd.concat([train_clean, val_clean], ignore_index=True, sort=False).tail(CONTEXT_STEPS)
        test_feat = build_features_with_context(
            df_history=tv_hist,
            df_current=test_clean,
            target_col=target_col,
            memory_threshold=float(args.memory_threshold),
            base_cols=base_cols_station,
        )

        train_parts.append(train_feat)
        val_parts.append(val_feat)
        test_parts.append(test_feat)

        stop_if_memory_high(float(args.memory_threshold))

    logging.info("Concatenating station splits...")
    train = pd.concat(train_parts, ignore_index=True, sort=False).sort_values(["station", "timestamp"]).reset_index(drop=True)
    val = pd.concat(val_parts, ignore_index=True, sort=False).sort_values(["station", "timestamp"]).reset_index(drop=True)
    test = pd.concat(test_parts, ignore_index=True, sort=False).sort_values(["station", "timestamp"]).reset_index(drop=True)

    # Optional but strongly recommended:
    # Stabilize the schema across splits using TRAIN as the source of truth.
    #
    # Why this matters:
    # - If a sensor feature is fully missing in val/test for all stations, and we
    #   infer base variables per-segment, val/test can end up missing entire sets
    #   of engineered columns (lags/rollmeans). That mismatch hurt RF/XGB badly.
    # - Using train as the schema source avoids using "future-only" columns and
    #   keeps evaluation honest.
    if bool(args.schema_stable):
        desired_cols = list(train.columns)

        # Ensure the common identifiers and target appear first for readability.
        front = [c for c in ("station", "timestamp", target_col, "source_file", "source_sheet") if c in desired_cols]
        rest = [c for c in desired_cols if c not in set(front)]
        desired_cols = front + rest

        # Reindex val/test:
        # - Missing columns are created as NaN.
        # - Extra columns (present only in val/test) are dropped (train-based schema).
        val = val.reindex(columns=desired_cols)
        test = test.reindex(columns=desired_cols)

        schema_path = Path(args.schema_json).resolve()
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        schema_payload = {
            "target": target_col,
            "columns": desired_cols,
            "lags": LAGS,
            "roll_windows": ROLL_WINDOWS,
            "per_station_base_cols": per_station_base_cols,
        }
        schema_path.write_text(json.dumps(schema_payload, indent=2), encoding="utf-8")
        logging.info("Saved feature schema JSON: %s", schema_path)

    # Normalization: fit scaler on train only, apply to all.
    # We scale only feature columns (exclude meta + target). This keeps target in
    # original units for easy interpretation and correct metric computation.
    feature_cols = [c for c in train.columns if c not in ("station", "timestamp", target_col, "source_file", "source_sheet")]
    logging.info("Fitting standard scaler on %d feature columns (train only)...", len(feature_cols))
    scaler = fit_standard_scaler(train, feature_cols=feature_cols)
    save_scaler_params(scaler_path, scaler)

    train = apply_standard_scaler(train, scaler, feature_cols=feature_cols)
    val = apply_standard_scaler(val, scaler, feature_cols=feature_cols)
    test = apply_standard_scaler(test, scaler, feature_cols=feature_cols)

    # 8) Save outputs.
    logging.info("Saving train/val/test feature parquet files...")
    save_parquet(train, out_train)
    save_parquet(val, out_val)
    save_parquet(test, out_test)

    logging.info(
        "Done. train=%d val=%d test=%d | scaler=%s",
        train.shape[0],
        val.shape[0],
        test.shape[0],
        scaler_path,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
