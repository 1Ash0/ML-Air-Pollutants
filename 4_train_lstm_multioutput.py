from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


META_COLS: Tuple[str, ...] = ("station", "timestamp", "source_file", "source_sheet")


def setup_logging(level: str) -> None:
    """Configure logging.

    Args:
        level: Logging level name (e.g., "INFO").
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def resolve_from_archive(filename: str) -> Optional[Path]:
    """Locate an archived copy of `filename` under `archive/` (best effort)."""
    archive_root = Path("archive")
    if not archive_root.exists():
        return None
    matches = list(archive_root.rglob(filename))
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def resolve_input_path(path: Path, archive_filename: str) -> Path:
    """Resolve a path; if missing, fall back to `archive/`."""
    if path.exists():
        return path
    alt = resolve_from_archive(archive_filename)
    if alt is None:
        raise FileNotFoundError(f"Missing input: {path} (and no archive/{archive_filename} found)")
    logging.warning("Input not found at %s; using archived copy: %s", path, alt)
    return alt


def load_parquet(path: Path) -> pd.DataFrame:
    """Load parquet into a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet: {path}")
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def available_default_targets(df: pd.DataFrame) -> List[str]:
    """Choose a sensible default set of pollutant targets present in the dataset."""
    preferred = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3", "NO", "NOX", "BENZ"]
    present = [c for c in preferred if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not present:
        raise ValueError("No default pollutant targets found in input dataframe.")
    return present


def add_horizon_targets(df: pd.DataFrame, targets: Sequence[str], horizon_steps: int) -> pd.DataFrame:
    """Create forecasting labels y_target(t+h) per station.

    Args:
        df: Split DataFrame containing station + pollutant columns.
        targets: Raw pollutant column names.
        horizon_steps: Horizon in 15-min steps (h>=1).

    Returns:
        DataFrame with extra columns: y_{target}_t+{h}.
    """
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be >= 1.")
    if "station" not in df.columns:
        raise ValueError("Expected 'station' for station-safe shifting.")
    out = df.copy()
    g = out.groupby("station", sort=False)
    for t in targets:
        out[f"y_{t}_t+{horizon_steps}"] = g[t].shift(-horizon_steps)
    return out


def drop_rows_with_missing_targets(df: pd.DataFrame, y_cols: Sequence[str]) -> pd.DataFrame:
    """Drop rows with missing y labels."""
    return df.dropna(subset=list(y_cols)).copy()


def drop_all_nan_feature_columns(train_df: pd.DataFrame, feature_cols: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Drop feature columns that are entirely missing in TRAIN."""
    kept: List[str] = []
    dropped: List[str] = []
    for c in feature_cols:
        if c not in train_df.columns:
            dropped.append(str(c))
            continue
        series = pd.to_numeric(train_df[c], errors="coerce")
        if series.notna().any():
            kept.append(str(c))
        else:
            dropped.append(str(c))
    return kept, dropped


def get_feature_columns(df: pd.DataFrame, exclude_cols: Sequence[str]) -> List[str]:
    """Select numeric feature columns, excluding identifiers and (raw) target columns.

    Leakage control:
      For a forecasting horizon model, we exclude the raw pollutant target columns so the
      network cannot trivially copy "current PM2.5" into "PM2.5(t+15min)".
      The engineered lag/rolling features remain and encode the past safely.
    """
    exclude = set(META_COLS) | set(exclude_cols)
    cols: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(str(c))
    return cols


@dataclass(frozen=True)
class Scalers:
    """Preprocessing objects saved for reproducibility."""

    feature_cols: List[str]
    dropped_feature_cols: List[str]
    x_imputer: object
    x_scaler: object
    y_scaler: object


def fit_scalers(train_df: pd.DataFrame, feature_cols: Sequence[str], y_cols: Sequence[str]) -> Scalers:
    """Fit imputers/scalers strictly on the training split.

    Args:
        train_df: Training DataFrame.
        feature_cols: Candidate feature columns.
        y_cols: Target columns (multi-output).

    Returns:
        Fitted scalers bundle.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    kept_cols, dropped_cols = drop_all_nan_feature_columns(train_df, feature_cols)
    if not kept_cols:
        raise ValueError("All feature columns are empty (all-NaN) in TRAIN after filtering.")

    x_train = train_df.loc[:, kept_cols].to_numpy(dtype=np.float32, copy=False)
    y_train = train_df.loc[:, list(y_cols)].to_numpy(dtype=np.float32, copy=False)

    x_imputer = SimpleImputer(strategy="median")
    x_train_imp = x_imputer.fit_transform(x_train)

    x_scaler = StandardScaler(with_mean=True, with_std=True)
    x_scaler.fit(x_train_imp)

    # One scaler for the full target matrix: each column gets its own mean/std.
    y_scaler = StandardScaler(with_mean=True, with_std=True)
    y_scaler.fit(y_train)

    return Scalers(
        feature_cols=list(kept_cols),
        dropped_feature_cols=list(dropped_cols),
        x_imputer=x_imputer,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
    )


def transform_xy(df: pd.DataFrame, scalers: Scalers, y_cols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Transform features and targets using pre-fit scalers.

    Returns:
        (X, Y) as float32 arrays:
          - X shape: (n_rows, n_features)
          - Y shape: (n_rows, n_targets)
    """
    x = df.loc[:, scalers.feature_cols].to_numpy(dtype=np.float32, copy=False)
    x_imp = scalers.x_imputer.transform(x)
    x_scaled = scalers.x_scaler.transform(x_imp).astype(np.float32, copy=False)

    y = df.loc[:, list(y_cols)].to_numpy(dtype=np.float32, copy=False)
    y_scaled = scalers.y_scaler.transform(y).astype(np.float32, copy=False)
    return x_scaled, y_scaled


def filter_non_finite_rows(
    df: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    split_name: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Drop rows where any feature/target is NaN/inf after preprocessing.

    Why this matters:
      Even a single NaN inside an LSTM input window can propagate and turn the
      loss into NaN, causing the entire training run to collapse.
    """
    if x.shape[0] != y.shape[0] or x.shape[0] != len(df):
        raise ValueError(f"{split_name}: misaligned shapes df={len(df)} x={x.shape} y={y.shape}")
    mask = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    dropped = int((~mask).sum())
    if dropped > 0:
        logging.warning("%s: dropping %d non-finite rows (NaN/inf) after scaling.", split_name, dropped)
    df_f = df.loc[mask].copy()
    x_f = x[mask]
    y_f = y[mask]
    return df_f, x_f, y_f


def build_station_dataset(
    df: pd.DataFrame,
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    seq_len: int,
    horizon_steps: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> "tf.data.Dataset":
    """Build station-safe sliding window datasets with an explicit forecast horizon.

    Tensor shapes (academic requirement):
      - Station features: X_station -> (n_timesteps, n_features)
      - Station targets:  Y_station -> (n_timesteps, n_targets)
      - Batches:
          X_batch -> (batch, timesteps=seq_len, features=n_features)
          Y_batch -> (batch, n_targets)

    Label alignment (no leakage):
      For each sequence ending at time t (inclusive):
        input window = X[t-seq_len+1 : t+1]
        label        = Y[t + horizon_steps]

      We implement this using `timeseries_dataset_from_array` with:
        data    = X[:-h]
        targets = Y[(h-1):]
      so that the dataset returns targets at index (i+seq_len), which maps to Y[t+h].
    """
    import tensorflow as tf

    if "station" not in df.columns:
        raise ValueError("Expected 'station' in df for station-safe sequencing.")
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be >= 1.")

    stations = df["station"].astype(str).to_numpy()
    datasets: List["tf.data.Dataset"] = []
    rng = np.random.default_rng(seed)

    for st in np.unique(stations):
        mask = stations == st
        x_st = x_scaled[mask]
        y_st = y_scaled[mask]
        n = int(x_st.shape[0])
        if n < (seq_len + horizon_steps + 5):
            continue

        # data: (n-h, n_features)
        # targets: (n-(h-1), n_targets)
        data = x_st[: n - horizon_steps]
        targets = y_st[(horizon_steps - 1) :]

        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=targets,
            sequence_length=int(seq_len),
            sequence_stride=1,
            sampling_rate=1,
            shuffle=bool(shuffle),
            seed=int(rng.integers(0, 2**31 - 1)),
            batch_size=int(batch_size),
        )
        datasets.append(ds)

    if not datasets:
        raise ValueError("No station had enough contiguous rows to create sequences.")

    out = datasets[0]
    for ds in datasets[1:]:
        out = out.concatenate(ds)

    # Cache/prefetch improves throughput on CPU; safe for moderate-sized datasets.
    return out.prefetch(tf.data.AUTOTUNE)


def build_model(seq_len: int, n_features: int, n_targets: int, lr: float) -> "tf.keras.Model":
    """Build a minimal multi-output LSTM model.

    Architecture:
      Input(shape=(T, n_features))
      LSTM(32, dropout=0.1)
      Dense(16, relu)
      Dense(n_targets)
    """
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(int(seq_len), int(n_features))),
            tf.keras.layers.LSTM(
                32,
                dropout=0.1,
                recurrent_dropout=0.0,
                return_sequences=False,
            ),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(int(n_targets)),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(lr)), loss="mse")
    return model


def metric_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def metric_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE."""
    return float(np.mean(np.abs(y_true - y_pred)))


def metric_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R^2 score."""
    denom = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if denom == 0.0:
        return float("nan")
    num = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - (num / denom)


def evaluate_multi(y_true: np.ndarray, y_pred: np.ndarray, target_names: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Per-target metrics."""
    out: Dict[str, Dict[str, float]] = {}
    for i, t in enumerate(target_names):
        yt = np.asarray(y_true[:, i], dtype=np.float32)
        yp = np.asarray(y_pred[:, i], dtype=np.float32)
        out[str(t)] = {"rmse": metric_rmse(yt, yp), "mae": metric_mae(yt, yp), "r2": metric_r2(yt, yp)}
    return out


def flatten_predictions(ds: "tf.data.Dataset", model: "tf.keras.Model") -> Tuple[np.ndarray, np.ndarray]:
    """Run model inference and return (y_true, y_pred) in scaled space."""
    import tensorflow as tf

    y_true_batches: List[np.ndarray] = []
    y_pred_batches: List[np.ndarray] = []

    for x_batch, y_batch in ds:
        pred = model(x_batch, training=False)
        y_true_batches.append(tf.convert_to_tensor(y_batch).numpy())
        y_pred_batches.append(tf.convert_to_tensor(pred).numpy())

    y_true = np.concatenate(y_true_batches, axis=0).astype(np.float32, copy=False)
    y_pred = np.concatenate(y_pred_batches, axis=0).astype(np.float32, copy=False)
    return y_true, y_pred


def training_has_nan(history: Dict[str, List[float]]) -> bool:
    """Return True if training history contains NaN."""
    for k, v in history.items():
        if not isinstance(v, list):
            continue
        arr = np.asarray(v, dtype=np.float32)
        if np.isnan(arr).any():
            return True
    return False


def append_multioutput_metrics_csv(
    out_csv: Path,
    model_name: str,
    station: str,
    metrics: Dict[str, Dict[str, float]],
    n_rows: int,
) -> None:
    """Append per-target metrics rows to `artifacts/multioutput_metrics.csv`."""
    rows: List[Dict[str, object]] = []
    for t, m in metrics.items():
        rows.append({"station": station, "model": model_name, "target": t, **m, "n_rows": int(n_rows)})
    df_out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        df_prev = pd.read_csv(out_csv)
        df_out = pd.concat([df_prev, df_out], ignore_index=True)
    df_out.to_csv(out_csv, index=False)
    logging.info("Updated: %s (+%d rows)", out_csv, len(rows))


def upsert_metrics_for_model(
    out_csv: Path,
    model_name: str,
    rows: List[Dict[str, object]],
) -> None:
    """Replace any existing rows for `model_name` then append fresh results.

    This prevents stale/blank rows from older partial runs from corrupting plots.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(rows)
    if out_csv.exists():
        df_prev = pd.read_csv(out_csv)
        df_prev = df_prev.loc[df_prev["model"].astype(str) != str(model_name)].copy()
        df_new = pd.concat([df_prev, df_new], ignore_index=True)
    df_new.to_csv(out_csv, index=False)
    logging.info("Upserted model=%s into %s (+%d rows)", model_name, out_csv, len(rows))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a multi-output LSTM for multi-pollutant forecasting.")
    parser.add_argument("--train", type=str, default=str(Path("artifacts") / "features_train_v4.parquet"))
    parser.add_argument("--val", type=str, default=str(Path("artifacts") / "features_val_v4.parquet"))
    parser.add_argument("--test", type=str, default=str(Path("artifacts") / "features_test_v4.parquet"))
    parser.add_argument("--targets", type=str, default="", help="Comma-separated raw pollutant targets (default autodetect).")
    parser.add_argument("--horizon-steps", type=int, default=1, help="Forecast horizon in 15-min steps (default 1).")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--reduce-lr-on-plateau", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--out-model", type=str, default=str(Path("artifacts") / "model_multi_lstm.keras"))
    parser.add_argument("--out-scalers", type=str, default=str(Path("artifacts") / "multi_lstm_scalers.pkl"))
    parser.add_argument("--out-metrics-json", type=str, default=str(Path("artifacts") / "multi_lstm_metrics.json"))
    parser.add_argument("--out-metrics-csv", type=str, default=str(Path("artifacts") / "multioutput_metrics.csv"))
    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        import tensorflow as tf  # noqa: F401
    except Exception as e:
        logging.error("TensorFlow import failed: %s", e)
        return 2

    train_path = resolve_input_path(Path(args.train), "features_train_v4.parquet")
    val_path = resolve_input_path(Path(args.val), "features_val_v4.parquet")
    test_path = resolve_input_path(Path(args.test), "features_test_v4.parquet")

    logging.info("Loading train/val/test...")
    df_train = load_parquet(train_path)
    df_val = load_parquet(val_path)
    df_test = load_parquet(test_path)

    targets = [t.strip() for t in args.targets.split(",") if t.strip()] if args.targets else available_default_targets(df_train)
    horizon = int(args.horizon_steps)
    seq_len = int(args.seq_len)
    batch = int(args.batch_size)
    epochs = int(args.epochs)

    logging.info("Targets=%s | horizon_steps=%s (+%d minutes) | seq_len=%d", targets, horizon, horizon * 15, seq_len)

    # Create explicit forecasting labels (t+h) and drop rows without labels.
    y_cols = [f"y_{t}_t+{horizon}" for t in targets]
    df_train = drop_rows_with_missing_targets(add_horizon_targets(df_train, targets=targets, horizon_steps=horizon), y_cols=y_cols)
    df_val = drop_rows_with_missing_targets(add_horizon_targets(df_val, targets=targets, horizon_steps=horizon), y_cols=y_cols)
    df_test = drop_rows_with_missing_targets(add_horizon_targets(df_test, targets=targets, horizon_steps=horizon), y_cols=y_cols)

    # Exclude raw targets from features to avoid trivial copying for short-horizon forecasts.
    feature_cols = get_feature_columns(df_train, exclude_cols=list(targets) + list(y_cols))
    kept_cols, dropped_cols = drop_all_nan_feature_columns(df_train, feature_cols)
    feature_cols = kept_cols

    logging.info("Fitting scalers on train only... features=%d", len(feature_cols))
    scalers = fit_scalers(df_train, feature_cols=feature_cols, y_cols=y_cols)

    x_train, y_train = transform_xy(df_train, scalers=scalers, y_cols=y_cols)
    x_val, y_val = transform_xy(df_val, scalers=scalers, y_cols=y_cols)
    x_test, y_test = transform_xy(df_test, scalers=scalers, y_cols=y_cols)

    # Safety: remove any NaN/inf rows after preprocessing so windows do not contain non-finite values.
    df_train, x_train, y_train = filter_non_finite_rows(df_train, x_train, y_train, split_name="TRAIN")
    df_val, x_val, y_val = filter_non_finite_rows(df_val, x_val, y_val, split_name="VAL")
    df_test, x_test, y_test = filter_non_finite_rows(df_test, x_test, y_test, split_name="TEST")

    logging.info("Building station-safe datasets... seq_len=%d horizon=%d batch=%d", seq_len, horizon, batch)
    ds_train = build_station_dataset(
        df=df_train,
        x_scaled=x_train,
        y_scaled=y_train,
        seq_len=seq_len,
        horizon_steps=horizon,
        batch_size=batch,
        shuffle=True,
        seed=42,
    )
    ds_val = build_station_dataset(
        df=df_val,
        x_scaled=x_val,
        y_scaled=y_val,
        seq_len=seq_len,
        horizon_steps=horizon,
        batch_size=batch,
        shuffle=False,
        seed=42,
    )
    ds_test = build_station_dataset(
        df=df_test,
        x_scaled=x_test,
        y_scaled=y_test,
        seq_len=seq_len,
        horizon_steps=horizon,
        batch_size=batch,
        shuffle=False,
        seed=42,
    )

    # element_spec shapes are dynamic in batch/time; we log expected meaning.
    logging.info("ds_train element_spec=%s", ds_train.element_spec)
    logging.info("Expected tensors: X=(batch, timesteps=%d, features=%d) | Y=(batch, targets=%d)", seq_len, len(scalers.feature_cols), len(y_cols))

    logging.info("Building model...")
    model = build_model(seq_len=seq_len, n_features=len(scalers.feature_cols), n_targets=len(y_cols), lr=float(args.learning_rate))
    model.summary(print_fn=lambda s: logging.info(s))

    callbacks: List[object] = []
    import tensorflow as tf

    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True))
    if args.reduce_lr_on_plateau:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, verbose=1))

    logging.info("Training...")
    history = model.fit(ds_train, validation_data=ds_val, epochs=epochs, callbacks=callbacks, verbose=2)

    if training_has_nan(history.history):
        logging.error(
            "Training produced NaN loss/val_loss. This is almost always caused by non-finite inputs/labels "
            "inside windows. Re-run after inspecting preprocessing; the script already filtered non-finite rows, "
            "so remaining issues may come from upstream feature generation."
        )
        return 3

    logging.info("Evaluating on test...")
    y_true_scaled, y_pred_scaled = flatten_predictions(ds_test, model)

    # Inverse-transform to original units.
    y_true = scalers.y_scaler.inverse_transform(y_true_scaled)
    y_pred = scalers.y_scaler.inverse_transform(y_pred_scaled)

    test_metrics = evaluate_multi(y_true, y_pred, target_names=y_cols)

    # Per-station test metrics (requires re-running per station dataset to keep alignment strict).
    metrics_by_station: Dict[str, Dict[str, Dict[str, float]]] = {}
    for st in sorted(df_test["station"].astype(str).unique()):
        df_s = df_test.loc[df_test["station"].astype(str) == st].copy()
        if len(df_s) < (seq_len + horizon + 10):
            continue
        x_s = x_test[df_test["station"].astype(str).to_numpy() == st]
        y_s = y_test[df_test["station"].astype(str).to_numpy() == st]
        ds_s = build_station_dataset(
            df=df_s,
            x_scaled=x_s,
            y_scaled=y_s,
            seq_len=seq_len,
            horizon_steps=horizon,
            batch_size=batch,
            shuffle=False,
            seed=42,
        )
        yt_s, yp_s = flatten_predictions(ds_s, model)
        yt_s = scalers.y_scaler.inverse_transform(yt_s)
        yp_s = scalers.y_scaler.inverse_transform(yp_s)
        metrics_by_station[st] = evaluate_multi(yt_s, yp_s, target_names=y_cols)

    # Upsert into the shared multioutput metrics CSV so `7_multioutput_plots.py` picks it up.
    out_csv = Path(args.out_metrics_csv)
    csv_rows: List[Dict[str, object]] = []
    for t, m in test_metrics.items():
        csv_rows.append({"station": "ALL", "model": "multi_lstm", "target": t, **m, "n_rows": int(y_true.shape[0])})
    for st, metrics_st in metrics_by_station.items():
        for t, m in metrics_st.items():
            csv_rows.append({"station": str(st), "model": "multi_lstm", "target": t, **m, "n_rows": 0})
    upsert_metrics_for_model(out_csv, model_name="multi_lstm", rows=csv_rows)

    # Persist model + scalers.
    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_model)
    logging.info("Saved model: %s", out_model)

    out_scalers = Path(args.out_scalers)
    import joblib

    joblib.dump(
        {
            "feature_cols": scalers.feature_cols,
            "dropped_feature_cols": scalers.dropped_feature_cols,
            "x_imputer": scalers.x_imputer,
            "x_scaler": scalers.x_scaler,
            "y_scaler": scalers.y_scaler,
            "targets": targets,
            "target_cols": y_cols,
            "horizon_steps": horizon,
        },
        out_scalers,
    )
    logging.info("Saved scalers: %s", out_scalers)

    out_json = Path(args.out_metrics_json)
    out_json.write_text(
        json.dumps(
            {
                "targets": targets,
                "target_cols": y_cols,
                "seq_len": seq_len,
                "horizon_steps": horizon,
                "n_features_used": len(scalers.feature_cols),
                "dropped_feature_cols": scalers.dropped_feature_cols,
                "tensor_shapes": {
                    "x_batch": "(batch, timesteps=seq_len, features=n_features_used)",
                    "y_batch": "(batch, n_targets)",
                },
                "history": {k: [float(x) for x in v] for k, v in history.history.items()},
                "test_metrics": test_metrics,
                "test_metrics_by_station": metrics_by_station,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logging.info("Saved metrics JSON: %s", out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
