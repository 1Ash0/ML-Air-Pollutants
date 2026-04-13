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


TARGET_DEFAULT: str = "PM2.5"
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


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file into a DataFrame.

    Args:
        path: Parquet path.

    Returns:
        Loaded DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet: {path}")
    return pd.read_parquet(path)


def select_feature_columns(df: pd.DataFrame, target: str) -> List[str]:
    """Select numeric feature columns for LSTM.

    Important (leakage control):
      We exclude the raw target column `PM2.5` from inputs, because the model is
      trained to predict it. The engineered lag/rolling features are already
      leakage-safe by construction and remain usable.

    Args:
        df: Input DataFrame.
        target: Target column name.

    Returns:
        List of numeric feature column names.
    """
    exclude = set(META_COLS) | {target}
    cols: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def drop_all_nan_feature_columns(train_df: pd.DataFrame, feature_cols: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Drop feature columns that are entirely missing in TRAIN.

    Why:
      `SimpleImputer(strategy="median")` cannot compute a statistic for columns
      that are all-NaN. sklearn will otherwise *drop* those columns implicitly,
      which makes it harder to report the true tensor shapes.

    Args:
        train_df: Training DataFrame.
        feature_cols: Candidate feature columns.

    Returns:
        (kept_cols, dropped_cols)
    """
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


@dataclass(frozen=True)
class Scalers:
    """Preprocessing objects saved for reproducibility."""

    feature_cols: List[str]
    dropped_feature_cols: List[str]
    x_imputer: object
    x_scaler: object
    y_scaler: object


def fit_scalers(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target: str,
) -> Scalers:
    """Fit scalers strictly on the training split.

    Academic / leakage note:
      - All statistics (imputation medians, feature scaling mean/std, and target
        scaling mean/std) are computed on TRAIN only.
      - The same transforms are then applied to val/test.

    Args:
        train_df: Training DataFrame.
        feature_cols: Feature columns.
        target: Target column.

    Returns:
        Fitted scalers bundle.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    kept_cols, dropped_cols = drop_all_nan_feature_columns(train_df, feature_cols)
    if not kept_cols:
        raise ValueError("All feature columns are empty (all-NaN) in TRAIN after filtering.")

    x_train = train_df.loc[:, kept_cols].to_numpy(dtype=np.float32, copy=False)
    y_train = train_df.loc[:, target].to_numpy(dtype=np.float32, copy=False).reshape(-1, 1)

    # 1) Feature imputation (median) then standardization.
    x_imputer = SimpleImputer(strategy="median")
    x_train_imp = x_imputer.fit_transform(x_train)

    x_scaler = StandardScaler(with_mean=True, with_std=True)
    x_scaler.fit(x_train_imp)

    # 2) Separate target scaling improves neural optimization stability.
    y_scaler = StandardScaler(with_mean=True, with_std=True)
    y_scaler.fit(y_train)

    return Scalers(
        feature_cols=list(kept_cols),
        dropped_feature_cols=list(dropped_cols),
        x_imputer=x_imputer,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
    )


def transform_xy(
    df: pd.DataFrame,
    scalers: Scalers,
    target: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform features and target using pre-fit scalers.

    Args:
        df: Input split DataFrame.
        scalers: Pre-fit scalers.
        target: Target column name.

    Returns:
        (X, y) as float32 arrays:
          - X shape: (n_rows, n_features)
          - y shape: (n_rows,) scaled target
    """
    x = df.loc[:, scalers.feature_cols].to_numpy(dtype=np.float32, copy=False)
    x_imp = scalers.x_imputer.transform(x)
    x_scaled = scalers.x_scaler.transform(x_imp).astype(np.float32, copy=False)

    y = df.loc[:, target].to_numpy(dtype=np.float32, copy=False).reshape(-1, 1)
    y_scaled = scalers.y_scaler.transform(y).astype(np.float32, copy=False).reshape(-1)
    return x_scaled, y_scaled


def build_station_datasets(
    df: pd.DataFrame,
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    target: str,
    seq_len: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> "tf.data.Dataset":
    """Create a tf.data.Dataset of sliding window sequences, grouped per station.

    Shape documentation (as required):
      - Raw scaled features for one station: X_station -> (n_timesteps, n_features)
      - Raw scaled targets for one station: y_station -> (n_timesteps,)
      - Sliding-window batches:
          X_batch -> (batch, timesteps=T, features=n_features)
          y_batch -> (batch, 1)

    Labeling scheme (1-step ahead forecast):
      For each station:
        input window  = X[t-seq_len : t]  (seq_len rows, ends at t-1)
        target label  = y[t]              (predict PM2.5 at time t)

      This is implemented by:
        data   = X[:-1]
        target = y[seq_len:]
      so the number of sequences is (n-1) - seq_len + 1 = n - seq_len.

    Args:
        df: Original split DataFrame (must include `station`, `timestamp`, target).
        x_scaled: Scaled X for the split (aligned to df rows).
        y_scaled: Scaled y for the split (aligned to df rows).
        target: Target column name.
        seq_len: Sequence length T.
        batch_size: Batch size.
        shuffle: Shuffle sequences within the dataset (train only).
        seed: RNG seed.

    Returns:
        A `tf.data.Dataset` yielding (X_batch, y_batch).
    """
    try:
        import tensorflow as tf  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing TensorFlow. Install it in your venv.\n"
            f"Import error: {e}"
        )

    if "station" not in df.columns:
        raise ValueError("Expected `station` column in features parquet.")
    if target not in df.columns:
        raise ValueError(f"Expected target column `{target}` in features parquet.")

    # We assume df is already sorted by station/timestamp.
    ds_all: Optional[tf.data.Dataset] = None
    n_features = int(x_scaled.shape[1])

    # Build per-station datasets so windows never cross station boundaries.
    for station, idx in df.groupby("station", sort=False).indices.items():
        station_indices = np.asarray(idx, dtype=np.int64)
        if station_indices.size <= seq_len + 1:
            continue

        x_s = x_scaled[station_indices]  # (n, n_features)
        y_s = y_scaled[station_indices]  # (n,)

        # Build next-step forecasting dataset.
        data = x_s[:-1]  # (n-1, n_features)
        targets = y_s[seq_len:]  # (n-seq_len,)

        # Each element produced:
        #   X_seq: (seq_len, n_features)
        #   y:     () scalar
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=targets,
            sequence_length=seq_len,
            sequence_stride=1,
            sampling_rate=1,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )

        # Ensure y has explicit shape (batch, 1).
        ds = ds.map(lambda xb, yb: (xb, tf.reshape(yb, (-1, 1))), num_parallel_calls=tf.data.AUTOTUNE)

        ds_all = ds if ds_all is None else ds_all.concatenate(ds)

    if ds_all is None:
        # Empty dataset
        x_spec = tf.TensorSpec(shape=(None, seq_len, n_features), dtype=tf.float32)
        y_spec = tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
        return tf.data.Dataset.from_generator(lambda: iter(()), output_signature=(x_spec, y_spec))

    # Performance: prefetch helps overlap CPU -> GPU/CPU execution.
    ds_all = ds_all.prefetch(tf.data.AUTOTUNE)
    return ds_all


def build_station_dataset_and_targets(
    df_station: pd.DataFrame,
    x_scaled_station: np.ndarray,
    y_scaled_station: np.ndarray,
    seq_len: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Tuple["tf.data.Dataset", np.ndarray]:
    """Build a station dataset plus its aligned y_true vector in scaled space.

    Shapes:
      - x_scaled_station: (n_timesteps, n_features)
      - y_scaled_station: (n_timesteps,)
      - dataset yields:
          X_batch: (batch, T=seq_len, n_features)
          y_batch: (batch, 1)
      - y_true_scaled returned here: (n_sequences,)
    """
    import tensorflow as tf  # type: ignore

    if x_scaled_station.shape[0] <= seq_len + 1:
        # Empty
        n_features = int(x_scaled_station.shape[1])
        x_spec = tf.TensorSpec(shape=(None, seq_len, n_features), dtype=tf.float32)
        y_spec = tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
        empty = tf.data.Dataset.from_generator(lambda: iter(()), output_signature=(x_spec, y_spec))
        return empty, np.array([], dtype=np.float32)

    data = x_scaled_station[:-1]  # (n-1, n_features)
    y_true_scaled = y_scaled_station[seq_len:]  # (n-seq_len,)

    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=y_true_scaled,
        sequence_length=seq_len,
        sequence_stride=1,
        sampling_rate=1,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    ds = ds.map(lambda xb, yb: (xb, tf.reshape(yb, (-1, 1))), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, np.asarray(y_true_scaled, dtype=np.float32)


def build_model(seq_len: int, n_features: int) -> "tf.keras.Model":
    """Build the required minimal LSTM architecture.

    Args:
        seq_len: Sequence length T.
        n_features: Number of input features.

    Returns:
        Compiled Keras model.
    """
    import tensorflow as tf  # type: ignore

    # Input tensor shape: (batch, timesteps=T, features=n_features)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(seq_len, n_features)),
            tf.keras.layers.LSTM(
                32,
                dropout=0.1,
                recurrent_dropout=0.0,
                return_sequences=False,
            ),  # output: (batch, 32)
            tf.keras.layers.Dense(16, activation="relu"),  # (batch, 16)
            tf.keras.layers.Dense(1),  # (batch, 1)
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="mse",
    )
    return model


def inverse_transform_target(y_scaled: np.ndarray, y_scaler: object) -> np.ndarray:
    """Inverse transform a scaled target vector back to original units."""
    y_scaled_2d = y_scaled.reshape(-1, 1).astype(np.float32, copy=False)
    y_inv = y_scaler.inverse_transform(y_scaled_2d).reshape(-1)
    return np.asarray(y_inv, dtype=np.float32)


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE/MAE/R²."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    r2 = float("nan") if denom == 0.0 else float(1.0 - (float(np.sum((y_true - y_pred) ** 2)) / denom))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def save_json(path: Path, payload: Dict[str, object]) -> None:
    """Save a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    """Train an LSTM model for PM2.5 forecasting."""
    parser = argparse.ArgumentParser(description="Phase D: Train LSTM on engineered features (leakage-safe).")
    parser.add_argument("--train", type=str, default=str(Path("artifacts") / "features_train_v4.parquet"))
    parser.add_argument("--val", type=str, default=str(Path("artifacts") / "features_val_v4.parquet"))
    parser.add_argument("--test", type=str, default=str(Path("artifacts") / "features_test_v4.parquet"))
    parser.add_argument("--target", type=str, default=TARGET_DEFAULT)
    parser.add_argument("--seq-len", type=int, default=96, help="T=96 for 15-min data (24 hours).")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-model", type=str, default=str(Path("artifacts") / "model_lstm.keras"))
    parser.add_argument("--out-scalers", type=str, default=str(Path("artifacts") / "lstm_scalers.pkl"))
    parser.add_argument("--out-metrics", type=str, default=str(Path("artifacts") / "lstm_metrics.json"))
    parser.add_argument(
        "--reduce-lr-on-plateau",
        action="store_true",
        help="Enable ReduceLROnPlateau callback (can slightly improve final val/test loss).",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    target = str(args.target)
    seq_len = int(args.seq_len)

    try:
        import tensorflow as tf  # type: ignore
    except Exception as e:
        logging.error("Missing TensorFlow. Install it in your venv. err=%s", e)
        return 10

    train_df = load_parquet(Path(args.train).resolve()).sort_values(["station", "timestamp"]).reset_index(drop=True)
    val_df = load_parquet(Path(args.val).resolve()).sort_values(["station", "timestamp"]).reset_index(drop=True)
    test_df = load_parquet(Path(args.test).resolve()).sort_values(["station", "timestamp"]).reset_index(drop=True)

    if target not in train_df.columns:
        logging.error("Target `%s` not found in train parquet.", target)
        return 2

    feature_cols = select_feature_columns(train_df, target=target)
    if not feature_cols:
        logging.error("No numeric feature columns found for LSTM.")
        return 3

    # Drop rows with missing target (LSTM training requires labels).
    train_df = train_df.dropna(subset=[target]).reset_index(drop=True)
    val_df = val_df.dropna(subset=[target]).reset_index(drop=True)
    test_df = test_df.dropna(subset=[target]).reset_index(drop=True)

    # 1) Normalization (fit strictly on train).
    logging.info("Fitting scalers on train only... features=%d", len(feature_cols))
    scalers = fit_scalers(train_df=train_df, feature_cols=feature_cols, target=target)

    x_train, y_train = transform_xy(train_df, scalers=scalers, target=target)
    x_val, y_val = transform_xy(val_df, scalers=scalers, target=target)
    x_test, y_test = transform_xy(test_df, scalers=scalers, target=target)

    # 2) Sequence generation (tf.data, station-safe).
    logging.info("Building station-safe datasets... seq_len=%d batch=%d", seq_len, int(args.batch_size))
    ds_train = build_station_datasets(
        df=train_df,
        x_scaled=x_train,
        y_scaled=y_train,
        target=target,
        seq_len=seq_len,
        batch_size=int(args.batch_size),
        shuffle=True,
        seed=int(args.seed),
    )
    ds_val = build_station_datasets(
        df=val_df,
        x_scaled=x_val,
        y_scaled=y_val,
        target=target,
        seq_len=seq_len,
        batch_size=int(args.batch_size),
        shuffle=False,
        seed=int(args.seed),
    )

    # Print tensor shape expectations to logs.
    logging.info("ds_train element_spec=%s", ds_train.element_spec)
    logging.info("ds_val element_spec=%s", ds_val.element_spec)

    # 3) Model architecture.
    model = build_model(seq_len=seq_len, n_features=int(x_train.shape[1]))
    model.summary(print_fn=lambda s: logging.info(s))

    # 4) Compile & train (early stopping).
    callbacks: List[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ]
    if bool(args.reduce_lr_on_plateau):
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5))

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=int(args.epochs),
        verbose=2,
        callbacks=callbacks,
    )

    # 5) Evaluate on test split (station-safe sequences).
    ds_test = build_station_datasets(
        df=test_df,
        x_scaled=x_test,
        y_scaled=y_test,
        target=target,
        seq_len=seq_len,
        batch_size=int(args.batch_size),
        shuffle=False,
        seed=int(args.seed),
    )

    # Collect predictions in scaled space, then inverse-transform to original units.
    y_pred_scaled = np.asarray(model.predict(ds_test, verbose=0), dtype=np.float32).reshape(-1)

    # Build y_true_scaled aligned with the concatenation order of `build_station_datasets()`.
    y_true_scaled_parts: List[np.ndarray] = []
    by_station_metrics: Dict[str, Dict[str, float]] = {}
    for station, idx in test_df.groupby("station", sort=False).indices.items():
        station_indices = np.asarray(idx, dtype=np.int64)
        x_s = x_test[station_indices]
        y_s = y_test[station_indices]
        ds_s, y_true_s_scaled = build_station_dataset_and_targets(
            df_station=test_df.iloc[station_indices],
            x_scaled_station=x_s,
            y_scaled_station=y_s,
            seq_len=seq_len,
            batch_size=int(args.batch_size),
            shuffle=False,
            seed=int(args.seed),
        )
        if y_true_s_scaled.size == 0:
            continue
        y_pred_s_scaled = np.asarray(model.predict(ds_s, verbose=0), dtype=np.float32).reshape(-1)
        # Metrics in original units for report clarity.
        y_true_s = inverse_transform_target(y_true_s_scaled, scalers.y_scaler)
        y_pred_s = inverse_transform_target(y_pred_s_scaled, scalers.y_scaler)
        by_station_metrics[str(station)] = evaluate_regression(y_true_s, y_pred_s)
        y_true_scaled_parts.append(y_true_s_scaled)

    y_true_scaled = np.concatenate(y_true_scaled_parts, axis=0) if y_true_scaled_parts else np.array([], dtype=np.float32)

    y_true = inverse_transform_target(y_true_scaled, scalers.y_scaler)
    y_pred = inverse_transform_target(y_pred_scaled, scalers.y_scaler)

    metrics = evaluate_regression(y_true, y_pred)
    logging.info("Test metrics (original units): %s", metrics)

    # Save artifacts.
    out_model = Path(args.out_model).resolve()
    out_scalers = Path(args.out_scalers).resolve()
    out_metrics = Path(args.out_metrics).resolve()

    out_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_model)

    try:
        import joblib
    except Exception as e:
        logging.error("Missing joblib for saving scalers. err=%s", e)
        return 11

    joblib.dump(
        {
            "feature_cols": scalers.feature_cols,
            "dropped_feature_cols": scalers.dropped_feature_cols,
            "x_imputer": scalers.x_imputer,
            "x_scaler": scalers.x_scaler,
            "y_scaler": scalers.y_scaler,
            "seq_len": seq_len,
            "target": target,
        },
        out_scalers,
    )
    save_json(
        out_metrics,
        {
            "target": target,
            "seq_len": seq_len,
            "n_features_requested": int(len(feature_cols)),
            "n_features_used": int(len(scalers.feature_cols)),
            "dropped_feature_cols": list(scalers.dropped_feature_cols),
            "tensor_shapes": {
                "x_batch": "(batch, timesteps=seq_len, features=n_features_used)",
                "y_batch": "(batch, 1)",
            },
            "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
            "test_metrics": metrics,
            "test_metrics_by_station": by_station_metrics,
        },
    )

    logging.info("Saved model: %s", out_model)
    logging.info("Saved scalers: %s", out_scalers)
    logging.info("Saved metrics: %s", out_metrics)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
