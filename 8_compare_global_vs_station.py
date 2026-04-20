from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


TARGET_DEFAULT: str = "PM2.5"
META_COLS: Tuple[str, ...] = ("station", "timestamp", "source_file", "source_sheet")


def setup_logging(level: str) -> None:
    """Configure logging."""
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
    """Load parquet."""
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def metric_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def metric_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE."""
    return float(np.mean(np.abs(y_true - y_pred)))


def metric_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R²."""
    denom = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if denom == 0.0:
        return float("nan")
    num = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - (num / denom)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression metrics."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return {"rmse": metric_rmse(y_true, y_pred), "mae": metric_mae(y_true, y_pred), "r2": metric_r2(y_true, y_pred)}


def evaluate_by_station(y_true: np.ndarray, y_pred: np.ndarray, station: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Station-wise metrics."""
    out: Dict[str, Dict[str, float]] = {}
    for s in np.unique(station):
        mask = station == s
        if int(np.sum(mask)) < 50:
            continue
        out[str(s)] = evaluate(y_true[mask], y_pred[mask])
    return out


def get_feature_columns(df: pd.DataFrame, target: str) -> List[str]:
    """Feature selection (matches Phase C logic)."""
    exclude = set(META_COLS) | {target, "station", "timestamp"}
    cols: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and not df[c].isna().all():
            cols.append(str(c))
    return cols


def build_xy(df: pd.DataFrame, feature_cols: Sequence[str], target: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, station) for evaluation."""
    clean = df.dropna(subset=[target]).copy()
    x = clean.reindex(columns=list(feature_cols)).to_numpy(dtype=np.float32, copy=False)
    y = clean[target].to_numpy(dtype=np.float32, copy=False)
    st = clean["station"].astype(str).to_numpy()
    return x, y, st


def train_global_ridge(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target: str,
    alphas: Sequence[float],
) -> Tuple[object, float, Dict[str, float]]:
    """Train a single global Ridge model (all stations pooled)."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x_train, y_train, _ = build_xy(train_df, feature_cols, target)
    x_val, y_val, _ = build_xy(val_df, feature_cols, target)

    best_alpha: Optional[float] = None
    best_model: Optional[object] = None
    best_metrics: Dict[str, float] = {}
    best_rmse = float("inf")

    for a in alphas:
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("ridge", Ridge(alpha=float(a), fit_intercept=True, random_state=42)),
            ]
        )
        model.fit(x_train, y_train)
        pred = np.asarray(model.predict(x_val), dtype=np.float32)
        m = evaluate(y_val, pred)
        logging.info("Global Ridge alpha=%s | val_rmse=%.4f", a, m["rmse"])
        if float(m["rmse"]) < best_rmse:
            best_rmse = float(m["rmse"])
            best_alpha = float(a)
            best_model = model
            best_metrics = m

    if best_model is None or best_alpha is None:
        raise RuntimeError("Failed to train global Ridge.")

    # Refit on Train+Val with best alpha (more data, less overfitting risk).
    train_val = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    x_tv, y_tv, _ = build_xy(train_val, feature_cols, target)
    best_model.fit(x_tv, y_tv)
    return best_model, best_alpha, best_metrics


def train_global_xgb(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target: str,
    max_train_rows: int,
) -> Tuple[object, Dict[str, float]]:
    """Train a single global XGBoost model with early stopping.

    Early stopping uses Validation split and does not leak Test information.
    """
    from xgboost import XGBRegressor

    x_train, y_train, _ = build_xy(train_df, feature_cols, target)
    x_val, y_val, _ = build_xy(val_df, feature_cols, target)

    # Optional subsample for speed on laptops.
    if max_train_rows > 0 and x_train.shape[0] > max_train_rows:
        idx = np.random.default_rng(42).choice(np.arange(x_train.shape[0]), size=max_train_rows, replace=False)
        x_train = x_train[idx]
        y_train = y_train[idx]
        logging.info("Subsampled global XGB train => %d rows", x_train.shape[0])

    model = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    pred_val = np.asarray(model.predict(x_val), dtype=np.float32)
    val_metrics = evaluate(y_val, pred_val)
    return model, val_metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Comparative study: global (all-stations) model vs per-station vs routed (PM2.5)."
    )
    parser.add_argument("--train", type=str, default=str(Path("artifacts") / "features_train_v4.parquet"))
    parser.add_argument("--val", type=str, default=str(Path("artifacts") / "features_val_v4.parquet"))
    parser.add_argument("--test", type=str, default=str(Path("artifacts") / "features_test_v4.parquet"))
    parser.add_argument("--target", type=str, default=TARGET_DEFAULT)
    parser.add_argument("--per-station-json", type=str, default=str(Path("artifacts") / "classical_metrics_per_station.json"))
    parser.add_argument("--routed-json", type=str, default=str(Path("artifacts") / "classical_metrics_routed_v4.json"))
    parser.add_argument("--out-json", type=str, default=str(Path("artifacts") / "pm25_comparison.json"))
    parser.add_argument("--xgb-max-train-rows", type=int, default=150_000)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    train_path = resolve_input_path(Path(args.train), "features_train_v4.parquet")
    val_path = resolve_input_path(Path(args.val), "features_val_v4.parquet")
    test_path = resolve_input_path(Path(args.test), "features_test_v4.parquet")

    train_df = load_parquet(train_path)
    val_df = load_parquet(val_path)
    test_df = load_parquet(test_path)

    target = str(args.target)
    if target not in train_df.columns:
        raise ValueError(f"Target column not found: {target}")

    feature_cols = get_feature_columns(train_df, target=target)
    logging.info("n_features=%d", len(feature_cols))

    # Global models
    logging.info("Training GLOBAL Ridge (all stations pooled)...")
    ridge_model, best_alpha, ridge_val = train_global_ridge(
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        target=target,
        alphas=[0.1, 1.0, 10.0],
    )
    x_test, y_test, st_test = build_xy(test_df, feature_cols, target)
    pred_ridge = np.asarray(ridge_model.predict(x_test), dtype=np.float32)
    ridge_test = evaluate(y_test, pred_ridge)
    ridge_test_by_station = evaluate_by_station(y_test, pred_ridge, st_test)

    logging.info("Training GLOBAL XGB (all stations pooled)...")
    xgb_model, xgb_val = train_global_xgb(
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        target=target,
        max_train_rows=int(args.xgb_max_train_rows),
    )
    pred_xgb = np.asarray(xgb_model.predict(x_test), dtype=np.float32)
    xgb_test = evaluate(y_test, pred_xgb)
    xgb_test_by_station = evaluate_by_station(y_test, pred_xgb, st_test)

    # Existing strategies from artifacts
    per_station = json.loads(Path(args.per_station_json).read_text(encoding="utf-8"))
    routed = json.loads(Path(args.routed_json).read_text(encoding="utf-8"))

    out = {
        "target": target,
        "global_ridge": {
            "best_alpha": best_alpha,
            "val": ridge_val,
            "test": ridge_test,
            "test_by_station": ridge_test_by_station,
        },
        "global_xgb": {
            "val": xgb_val,
            "test": xgb_test,
            "test_by_station": xgb_test_by_station,
        },
        "per_station_models": per_station,
        "routed_models": routed,
        "interpretation_notes": [
            "Global models use more data but must fit a single set of parameters across stations.",
            "Per-station models reduce heterogeneity but risk overfitting; we mitigate via time-based splits and validation-based routing.",
            "Compare validation vs test metrics to judge overfitting; large gaps indicate overfit/underfit.",
        ],
    }
    out_path = Path(args.out_json)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logging.info("Wrote: %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

