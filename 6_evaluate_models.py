from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R2 score (1 - SSE/SST)."""
    denom = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if denom == 0.0:
        return float("nan")
    num = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - (num / denom)


def select_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    """Select numeric feature columns for non-sequence models."""
    exclude = {"station", "timestamp", "source_file", "source_sheet", target_col}
    cols: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def split_xy(df: pd.DataFrame, feature_cols: Sequence[str], target_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """Create X/y arrays."""
    x = df.loc[:, list(feature_cols)].to_numpy(dtype=np.float32, copy=False)
    y = df.loc[:, target_col].to_numpy(dtype=np.float32, copy=False)
    return x, y


def persistence_baseline(df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """Simple persistence baseline using the existing lag-1 feature if present.

    If `{target}_lag_1` exists (it should from Phase B), we use that as the
    prediction.
    """
    lag_col = f"{target_col}_lag_1"
    if lag_col not in df.columns:
        raise ValueError(f"Missing persistence lag column: {lag_col}")
    y_true = df[target_col].to_numpy(dtype=np.float32, copy=False)
    y_pred = df[lag_col].to_numpy(dtype=np.float32, copy=False)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def load_sklearn_model(path: Path) -> Tuple[object, List[str], str]:
    """Load a sklearn model bundle saved by 3_train_models_sklearn.py."""
    import joblib

    bundle = joblib.load(path)
    return bundle["model"], list(bundle["feature_cols"]), str(bundle["target"])


def load_xgb_model(model_path: Path, meta_path: Path) -> Tuple[object, List[str], str]:
    """Load XGBoost model and meta."""
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(model_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return booster, list(meta["feature_cols"]), str(meta["target"])


def eval_model_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard metrics (RMSE, MAE, R2)."""
    return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred), "r2": r2(y_true, y_pred)}


def main() -> int:
    """Evaluate all available models on the test split."""
    parser = argparse.ArgumentParser(description="Evaluate trained models on test set and write metrics.")
    parser.add_argument("--test", type=str, default=str(Path("artifacts") / "features_test.parquet"))
    parser.add_argument("--models-dir", type=str, default=str(Path("artifacts") / "models"))
    parser.add_argument("--out-metrics", type=str, default=str(Path("artifacts") / "metrics.csv"))
    parser.add_argument("--out-by-station", type=str, default=str(Path("artifacts") / "metrics_by_station.csv"))
    parser.add_argument("--target", type=str, default=TARGET_COL_DEFAULT)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    test_df = load_parquet(Path(args.test).resolve())
    target_col = str(args.target)

    if target_col not in test_df.columns:
        logging.error("Target column missing in test: %s", target_col)
        return 2

    models_dir = Path(args.models_dir).resolve()
    out_metrics = Path(args.out_metrics).resolve()
    out_by_station = Path(args.out_by_station).resolve()
    out_metrics.parent.mkdir(parents=True, exist_ok=True)

    metrics_rows: List[Dict[str, object]] = []
    station_rows: List[Dict[str, object]] = []

    # Baseline
    logging.info("Evaluating persistence baseline...")
    y_true_b, y_pred_b = persistence_baseline(test_df, target_col=target_col)
    base_metrics = eval_model_predictions(y_true_b, y_pred_b)
    metrics_rows.append({"model": "persistence", "station": "ALL", **base_metrics})

    # Per-station baseline
    for station, sdf in test_df.groupby("station", sort=False):
        try:
            yt, yp = persistence_baseline(sdf, target_col=target_col)
            m = eval_model_predictions(yt, yp)
            station_rows.append({"model": "persistence", "station": str(station), **m})
        except Exception:
            continue

    # sklearn models
    for name, filename in [("ridge", "ridge.joblib"), ("random_forest", "random_forest.joblib")]:
        model_path = models_dir / filename
        if not model_path.exists():
            logging.warning("Missing model: %s", model_path)
            continue

        logging.info("Evaluating %s...", name)
        model, feature_cols, target_saved = load_sklearn_model(model_path)
        if target_saved != target_col:
            logging.warning("Target mismatch for %s: saved=%s arg=%s", name, target_saved, target_col)

        # Drop NaNs for sklearn inference.
        df_eval = test_df.dropna(subset=[target_col] + feature_cols).reset_index(drop=True)
        x, y = split_xy(df_eval, feature_cols, target_col)
        y_pred = np.asarray(model.predict(x), dtype=np.float32)
        m_all = eval_model_predictions(y, y_pred)
        metrics_rows.append({"model": name, "station": "ALL", **m_all})

        for station, sdf in df_eval.groupby("station", sort=False):
            x_s, y_s = split_xy(sdf, feature_cols, target_col)
            y_pred_s = np.asarray(model.predict(x_s), dtype=np.float32)
            m_s = eval_model_predictions(y_s, y_pred_s)
            station_rows.append({"model": name, "station": str(station), **m_s})

    # XGBoost
    xgb_model_path = models_dir / "xgb.json"
    xgb_meta_path = models_dir / "xgb_meta.json"
    if xgb_model_path.exists() and xgb_meta_path.exists():
        try:
            import xgboost as xgb
        except Exception as e:
            logging.warning("xgboost not installed; skipping XGB eval. err=%s", e)
        else:
            logging.info("Evaluating xgboost...")
            booster, feature_cols, target_saved = load_xgb_model(xgb_model_path, xgb_meta_path)
            if target_saved != target_col:
                logging.warning("Target mismatch for xgb: saved=%s arg=%s", target_saved, target_col)

            df_eval = test_df.dropna(subset=[target_col] + feature_cols).reset_index(drop=True)
            x_arr, y_arr = split_xy(df_eval, feature_cols, target_col)
            dtest = xgb.DMatrix(x_arr, feature_names=list(feature_cols))
            y_pred = np.asarray(booster.predict(dtest), dtype=np.float32)
            m_all = eval_model_predictions(y_arr, y_pred)
            metrics_rows.append({"model": "xgboost", "station": "ALL", **m_all})

            for station, sdf in df_eval.groupby("station", sort=False):
                x_s, y_s = split_xy(sdf, feature_cols, target_col)
                d_s = xgb.DMatrix(x_s, feature_names=list(feature_cols))
                y_pred_s = np.asarray(booster.predict(d_s), dtype=np.float32)
                m_s = eval_model_predictions(y_s, y_pred_s)
                station_rows.append({"model": "xgboost", "station": str(station), **m_s})
    else:
        logging.warning("XGB model/meta missing; skipping xgboost eval.")

    # Write metrics
    logging.info("Writing metrics: %s", out_metrics)
    with out_metrics.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "station", "rmse", "mae", "r2"])
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)

    logging.info("Writing per-station metrics: %s", out_by_station)
    with out_by_station.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "station", "rmse", "mae", "r2"])
        writer.writeheader()
        for row in station_rows:
            writer.writerow(row)

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise

