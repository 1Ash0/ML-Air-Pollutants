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
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def metric_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def metric_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def metric_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if denom == 0.0:
        return float("nan")
    num = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - (num / denom)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return {"rmse": metric_rmse(y_true, y_pred), "mae": metric_mae(y_true, y_pred), "r2": metric_r2(y_true, y_pred)}


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


@dataclass(frozen=True)
class SklearnBundle:
    model: object
    feature_cols: List[str]


def load_joblib_bundle(path: Path) -> SklearnBundle:
    import joblib

    if not path.exists():
        raise FileNotFoundError(path)
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"Unexpected joblib bundle: {path}")
    feats = bundle.get("feature_cols", [])
    if not isinstance(feats, list) or not feats:
        raise ValueError(f"Missing feature_cols in {path}")
    return SklearnBundle(model=bundle["model"], feature_cols=[str(c) for c in feats])


@dataclass(frozen=True)
class XgbBundle:
    model: object
    feature_cols: List[str]


def load_xgb_bundle(model_path: Path, meta_path: Path) -> XgbBundle:
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing XGB files: {model_path} / {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feats = meta.get("feature_cols", [])
    if not isinstance(feats, list) or not feats:
        raise ValueError(f"Missing feature_cols in {meta_path}")
    from xgboost import XGBRegressor

    model = XGBRegressor()
    model.load_model(str(model_path))
    return XgbBundle(model=model, feature_cols=[str(c) for c in feats])


def station_model_paths(artifacts: Path, station: str) -> Dict[str, Path]:
    safe = str(station).replace(" ", "_").upper()
    return {
        "ridge": artifacts / f"model_ridge_{safe}.pkl",
        "rf": artifacts / f"model_rf_{safe}.pkl",
        "xgb": artifacts / f"model_xgb_{safe}.json",
        "xgb_meta": artifacts / f"model_xgb_{safe}_meta.json",
    }


@dataclass(frozen=True)
class LstmArtifacts:
    model: object
    feature_cols: List[str]
    x_imputer: object
    x_scaler: object
    y_scaler: object
    seq_len: int


def load_lstm_artifacts(model_path: Path, scalers_path: Path) -> LstmArtifacts:
    import joblib

    try:
        import tensorflow as tf  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Missing tensorflow for LSTM inference. err={e}")

    model = tf.keras.models.load_model(model_path)
    bundle = joblib.load(scalers_path)
    return LstmArtifacts(
        model=model,
        feature_cols=[str(c) for c in bundle["feature_cols"]],
        x_imputer=bundle["x_imputer"],
        x_scaler=bundle["x_scaler"],
        y_scaler=bundle["y_scaler"],
        seq_len=int(bundle.get("seq_len", 96)),
    )


def inverse_transform_target(y_scaled: np.ndarray, y_scaler: object) -> np.ndarray:
    y_scaled_2d = np.asarray(y_scaled, dtype=np.float32).reshape(-1, 1)
    y_inv = y_scaler.inverse_transform(y_scaled_2d).reshape(-1)
    return np.asarray(y_inv, dtype=np.float32)


def lstm_predict_split(
    df: pd.DataFrame,
    target: str,
    lstm: LstmArtifacts,
    batch_size: int,
) -> pd.DataFrame:
    """Predict LSTM on a split, station-safe, returning aligned timestamps.

    The first `seq_len` timestamps per station cannot be predicted (no full window).
    """
    import tensorflow as tf  # type: ignore

    parts: List[pd.DataFrame] = []
    for station, sdf in df.groupby("station", sort=False):
        sdf = sdf.sort_values("timestamp").dropna(subset=[target]).reset_index(drop=True)
        if sdf.shape[0] <= lstm.seq_len + 1:
            continue

        x = sdf.reindex(columns=lstm.feature_cols).to_numpy(dtype=np.float32, copy=False)
        x_imp = lstm.x_imputer.transform(x)
        x_scaled = lstm.x_scaler.transform(x_imp).astype(np.float32, copy=False)

        y = pd.to_numeric(sdf[target], errors="coerce").to_numpy(dtype=np.float32, copy=False).reshape(-1, 1)
        y_scaled = lstm.y_scaler.transform(y).astype(np.float32, copy=False).reshape(-1)

        data = x_scaled[:-1]
        y_true_scaled = y_scaled[lstm.seq_len :]
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=y_true_scaled,
            sequence_length=int(lstm.seq_len),
            sequence_stride=1,
            sampling_rate=1,
            batch_size=int(batch_size),
            shuffle=False,
        )
        pred_scaled = np.asarray(lstm.model.predict(ds, verbose=0), dtype=np.float32).reshape(-1)

        y_true = inverse_transform_target(y_true_scaled, lstm.y_scaler)
        pred = inverse_transform_target(pred_scaled, lstm.y_scaler)

        ts = pd.to_datetime(sdf["timestamp"], errors="coerce").to_numpy(copy=False)
        ts_aligned = ts[lstm.seq_len : lstm.seq_len + pred.shape[0]]

        parts.append(
            pd.DataFrame(
                {
                    "station": np.asarray([str(station)] * int(pred.shape[0])),
                    "timestamp": ts_aligned,
                    "y_true": y_true,
                    "pred_lstm": pred,
                }
            )
        )
    if not parts:
        return pd.DataFrame(columns=["station", "timestamp", "y_true", "pred_lstm"])
    return pd.concat(parts, ignore_index=True, sort=False).sort_values(["station", "timestamp"]).reset_index(drop=True)


def classical_predict_split_per_station(
    df: pd.DataFrame,
    artifacts: Path,
    target: str,
) -> pd.DataFrame:
    """Predict per-station Ridge/RF/XGB for all rows in df (target-nonnull)."""
    out_parts: List[pd.DataFrame] = []
    for station, sdf in df.groupby("station", sort=False):
        sdf = sdf.dropna(subset=[target]).sort_values("timestamp").copy()
        if sdf.empty:
            continue
        paths = station_model_paths(artifacts, str(station))
        ridge = load_joblib_bundle(paths["ridge"])
        rf = load_joblib_bundle(paths["rf"])
        xgb = load_xgb_bundle(paths["xgb"], paths["xgb_meta"])

        x_ridge = sdf.reindex(columns=ridge.feature_cols).to_numpy(dtype=np.float32, copy=False)
        x_rf = sdf.reindex(columns=rf.feature_cols).to_numpy(dtype=np.float32, copy=False)
        x_xgb = sdf.reindex(columns=xgb.feature_cols).to_numpy(dtype=np.float32, copy=False)

        pred_ridge = np.asarray(ridge.model.predict(x_ridge), dtype=np.float32)
        pred_rf = np.asarray(rf.model.predict(x_rf), dtype=np.float32)
        pred_xgb = np.asarray(xgb.model.predict(x_xgb), dtype=np.float32)

        out_parts.append(
            pd.DataFrame(
                {
                    "station": sdf["station"].astype(str).to_numpy(copy=False),
                    "timestamp": pd.to_datetime(sdf["timestamp"], errors="coerce").to_numpy(copy=False),
                    "y_true": pd.to_numeric(sdf[target], errors="coerce").to_numpy(dtype=np.float32, copy=False),
                    "pred_ridge": pred_ridge,
                    "pred_rf": pred_rf,
                    "pred_xgb": pred_xgb,
                }
            )
        )
    if not out_parts:
        return pd.DataFrame(columns=["station", "timestamp", "y_true", "pred_ridge", "pred_rf", "pred_xgb"])
    return pd.concat(out_parts, ignore_index=True, sort=False).sort_values(["station", "timestamp"]).reset_index(drop=True)


def compute_metrics_rows(split: str, df_preds: pd.DataFrame, pred_cols: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for station, sdf in [("ALL", df_preds)] + [(str(k), v) for k, v in df_preds.groupby("station", sort=False)]:
        y = sdf["y_true"].to_numpy(dtype=np.float32, copy=False)
        for col in pred_cols:
            p = sdf[col].to_numpy(dtype=np.float32, copy=False)
            m = evaluate(y, p)
            rows.append({"split": split, "station": station, "model": col.replace("pred_", ""), **m, "n_rows": int(sdf.shape[0])})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate viva-ready metrics for train/val/test.")
    parser.add_argument("--train", type=str, default=str(Path("artifacts") / "features_train_v4.parquet"))
    parser.add_argument("--val", type=str, default=str(Path("artifacts") / "features_val_v4.parquet"))
    parser.add_argument("--test", type=str, default=str(Path("artifacts") / "features_test_v4.parquet"))
    parser.add_argument("--target", type=str, default=TARGET_DEFAULT)
    parser.add_argument("--artifacts-dir", type=str, default=str(Path("artifacts")))
    parser.add_argument("--lstm-model", type=str, default=str(Path("artifacts") / "model_lstm.keras"))
    parser.add_argument("--lstm-scalers", type=str, default=str(Path("artifacts") / "lstm_scalers.pkl"))
    parser.add_argument("--lstm-batch-size", type=int, default=256)
    parser.add_argument("--out-csv", type=str, default=str(Path("artifacts") / "viva_metrics_splits.csv"))
    parser.add_argument("--out-notes", type=str, default=str(Path("artifacts") / "viva_metrics_notes.json"))
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    artifacts = Path(args.artifacts_dir).resolve()
    target = str(args.target)

    train_df = load_parquet(Path(args.train).resolve()).sort_values(["station", "timestamp"]).reset_index(drop=True)
    val_df = load_parquet(Path(args.val).resolve()).sort_values(["station", "timestamp"]).reset_index(drop=True)
    test_df = load_parquet(Path(args.test).resolve()).sort_values(["station", "timestamp"]).reset_index(drop=True)

    # LSTM predictions define the aligned evaluation window (drops first T steps per station).
    lstm = load_lstm_artifacts(Path(args.lstm_model).resolve(), Path(args.lstm_scalers).resolve())
    logging.info("Loaded LSTM: seq_len=%d n_features=%d", lstm.seq_len, len(lstm.feature_cols))

    rows_all: List[Dict[str, object]] = []
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        logging.info("Predicting split=%s ...", split_name)
        pred_classical = classical_predict_split_per_station(split_df, artifacts=artifacts, target=target)
        pred_lstm = lstm_predict_split(split_df, target=target, lstm=lstm, batch_size=int(args.lstm_batch_size))

        # Align all models on timestamps where LSTM is defined.
        eval_df = pred_classical.merge(pred_lstm[["station", "timestamp", "pred_lstm"]], on=["station", "timestamp"], how="inner")
        eval_df = eval_df.dropna(subset=["y_true", "pred_ridge", "pred_rf", "pred_xgb", "pred_lstm"]).reset_index(drop=True)
        if eval_df.empty:
            logging.warning("No aligned rows for split=%s; skipping.", split_name)
            continue

        rows_all.extend(compute_metrics_rows(split=split_name, df_preds=eval_df, pred_cols=["pred_ridge", "pred_rf", "pred_xgb", "pred_lstm"]))

    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_all).to_csv(out_csv, index=False)
    logging.info("Wrote: %s", out_csv)

    notes = {
        "target": target,
        "alignment": "All split metrics are computed on the intersection of timestamps where LSTM produces predictions (drops first seq_len steps per station).",
        "lstm_seq_len": int(lstm.seq_len),
        "models": ["ridge", "rf", "xgb", "lstm"],
    }
    out_notes = Path(args.out_notes).resolve()
    out_notes.parent.mkdir(parents=True, exist_ok=True)
    out_notes.write_text(json.dumps(notes, indent=2), encoding="utf-8")
    logging.info("Wrote: %s", out_notes)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise

