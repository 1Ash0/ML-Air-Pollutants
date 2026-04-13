from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


TARGET_COL_DEFAULT: str = "PM2.5"


def setup_logging(level: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_parquet(path: Path) -> pd.DataFrame:
    """Load parquet into a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet: {path}")
    return pd.read_parquet(path)


def select_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    """Select numeric feature columns for LSTM (exclude identifiers and target).

    Note:
      Phase B already scales feature columns; we keep that to simplify DL training.
    """
    exclude = {"station", "timestamp", "source_file", "source_sheet", target_col}
    cols: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


@dataclass(frozen=True)
class SequenceDataset:
    """Numpy arrays for sequence training."""

    x: np.ndarray  # (n_samples, seq_len, n_features)
    y: np.ndarray  # (n_samples,)


def build_sequences_per_station(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    seq_len: int,
) -> SequenceDataset:
    """Build leakage-safe sequences per station.

    For each station, we create samples where:
      X[t] = features at times [t-seq_len .. t-1]
      y[t] = target at time t

    This prevents leakage because the model never sees time t inputs when predicting y(t).

    Args:
        df: Input DataFrame sorted by station/timestamp.
        feature_cols: Feature columns (scaled).
        target_col: Target column.
        seq_len: Number of past steps to use.

    Returns:
        SequenceDataset with concatenated sequences across stations.
    """
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    for _, sdf in df.groupby("station", sort=False):
        sdf = sdf.sort_values("timestamp")
        sdf = sdf.dropna(subset=[target_col] + list(feature_cols)).reset_index(drop=True)
        if sdf.shape[0] <= seq_len:
            continue

        feat = sdf.loc[:, list(feature_cols)].to_numpy(dtype=np.float32, copy=False)
        targ = sdf.loc[:, target_col].to_numpy(dtype=np.float32, copy=False)

        n = feat.shape[0]
        # Build sequences with a simple sliding window.
        x_station = np.zeros((n - seq_len, seq_len, feat.shape[1]), dtype=np.float32)
        y_station = targ[seq_len:].copy()
        for i in range(seq_len, n):
            x_station[i - seq_len] = feat[i - seq_len : i]
        xs.append(x_station)
        ys.append(y_station)

    if not xs:
        return SequenceDataset(x=np.empty((0, seq_len, len(feature_cols)), dtype=np.float32), y=np.empty((0,), dtype=np.float32))

    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    return SequenceDataset(x=x_all, y=y_all)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    """Write JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    """Train a lightweight LSTM model."""
    parser = argparse.ArgumentParser(description="Train lightweight LSTM on engineered features.")
    parser.add_argument("--train", type=str, default=str(Path("artifacts") / "features_train.parquet"))
    parser.add_argument("--val", type=str, default=str(Path("artifacts") / "features_val.parquet"))
    parser.add_argument("--out-dir", type=str, default=str(Path("artifacts") / "models"))
    parser.add_argument("--target", type=str, default=TARGET_COL_DEFAULT)
    parser.add_argument("--seq-len", type=int, default=96, help="Sequence length in 15-min steps (96 = 24 hours).")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        import tensorflow as tf  # type: ignore
    except Exception as e:
        logging.error("Missing tensorflow. Install requirements_ml.txt. err=%s", e)
        return 4

    train_df = load_parquet(Path(args.train).resolve())
    val_df = load_parquet(Path(args.val).resolve())
    target_col = str(args.target)

    if target_col not in train_df.columns:
        logging.error("Target column missing in train: %s", target_col)
        return 2

    feature_cols = select_feature_columns(train_df, target_col=target_col)
    if not feature_cols:
        logging.error("No numeric feature columns found.")
        return 3

    train_df = train_df.sort_values(["station", "timestamp"]).reset_index(drop=True)
    val_df = val_df.sort_values(["station", "timestamp"]).reset_index(drop=True)

    seq_len = int(args.seq_len)
    logging.info("Building sequences: seq_len=%d features=%d", seq_len, len(feature_cols))
    train_seq = build_sequences_per_station(train_df, feature_cols, target_col, seq_len=seq_len)
    val_seq = build_sequences_per_station(val_df, feature_cols, target_col, seq_len=seq_len)

    if train_seq.x.shape[0] == 0 or val_seq.x.shape[0] == 0:
        logging.error("Not enough data to build sequences. train=%s val=%s", train_seq.x.shape, val_seq.x.shape)
        return 5

    logging.info("Train sequences: %s | Val sequences: %s", train_seq.x.shape, val_seq.x.shape)

    # Lightweight architecture for 24-hour deadline.
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(seq_len, len(feature_cols))),
            tf.keras.layers.LSTM(32, dropout=0.1, return_sequences=False),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

    callbacks: List[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ]

    history = model.fit(
        train_seq.x,
        train_seq.y,
        validation_data=(val_seq.x, val_seq.y),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        callbacks=callbacks,
        verbose=2,
    )

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "lstm.keras"
    meta_path = out_dir / "lstm_meta.json"

    model.save(model_path)
    save_json(
        meta_path,
        {
            "target": target_col,
            "feature_cols": list(feature_cols),
            "seq_len": seq_len,
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        },
    )

    logging.info("Saved LSTM model to %s", model_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise

