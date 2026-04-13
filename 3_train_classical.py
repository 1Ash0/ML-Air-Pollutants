from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


TARGET_DEFAULT: str = "PM2.5"


def setup_logging(level: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file.

    Args:
        path: Parquet file path.

    Returns:
        DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet: {path}")
    return pd.read_parquet(path)


def metric_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def metric_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAE."""
    return float(np.mean(np.abs(y_true - y_pred)))


def metric_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² (1 - SSE/SST)."""
    denom = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if denom == 0.0:
        return float("nan")
    num = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - (num / denom)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression metrics."""
    return {
        "rmse": metric_rmse(y_true, y_pred),
        "mae": metric_mae(y_true, y_pred),
        "r2": metric_r2(y_true, y_pred),
    }


def evaluate_by_station(y_true: np.ndarray, y_pred: np.ndarray, station: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compute regression metrics per station.

    This is a diagnostics helper: overall metrics can look "fine" while one
    station performs very poorly due to different sensor coverage or local
    emission patterns.

    Args:
        y_true: Ground truth array.
        y_pred: Prediction array.
        station: Station labels aligned with y arrays.

    Returns:
        Dict mapping station -> metrics dict.
    """
    out: Dict[str, Dict[str, float]] = {}
    if station.size == 0:
        return out
    for s in np.unique(station):
        mask = station == s
        if int(np.sum(mask)) < 10:
            continue
        out[str(s)] = evaluate(y_true[mask], y_pred[mask])
    return out


def get_feature_columns(df: pd.DataFrame, target: str) -> List[str]:
    """Select feature columns for classical models.

    We keep *all numeric engineered columns* (lags, rolling means, temporal encodings,
    meteorology) and exclude identifiers and the target.

    Args:
        df: Input DataFrame.
        target: Target column name.

    Returns:
        Feature column list.
    """
    exclude = {"station", "timestamp", "source_file", "source_sheet", target}
    cols: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            # Prune columns that are 100% NaN to avoid zero-row training sets
            if not df[c].isna().all():
                cols.append(c)
    return cols


def align_feature_columns(dfs: Sequence[pd.DataFrame], feature_cols: Sequence[str]) -> List[pd.DataFrame]:
    """Reindex all splits to the same feature columns (missing columns become NaN)."""
    out: List[pd.DataFrame] = []
    for df in dfs:
        out.append(df.reindex(columns=list(feature_cols)))
    return out


def subsample_by_station(df: pd.DataFrame, max_rows: int, random_state: int = 42) -> pd.DataFrame:
    """Subsample a dataframe while preserving station representation.

    Why:
      A global random sample can accidentally under-sample a "hard" station (e.g.,
      BHATAGAON) and make tree models look worse (or unstable) than they should.

    Strategy:
      - Allocate sample budget proportional to station row counts.
      - Sample within each station.
      - Return chronologically sorted rows (station, timestamp).

    Args:
        df: Input DataFrame with `station` and `timestamp`.
        max_rows: Maximum rows to keep. If <= 0, returns df unchanged.
        random_state: RNG seed.

    Returns:
        Subsampled DataFrame.
    """
    if max_rows <= 0 or df.shape[0] <= max_rows:
        return df
    if "station" not in df.columns:
        return df.sample(n=max_rows, random_state=random_state)

    parts: List[pd.DataFrame] = []
    counts = df["station"].value_counts()
    total = int(counts.sum())
    if total == 0:
        return df.sample(n=max_rows, random_state=random_state)

    # Allocate proportional quotas, but guarantee at least a small minimum.
    min_per_station = max(1000, int(max_rows * 0.02))
    remaining = max_rows
    quotas: Dict[str, int] = {}
    for station, cnt in counts.items():
        q = int(round((int(cnt) / total) * max_rows))
        q = max(min_per_station, q)
        q = min(int(cnt), q)
        quotas[str(station)] = q
    # If we overshoot, scale down uniformly.
    q_sum = int(sum(quotas.values()))
    if q_sum > max_rows:
        scale = max_rows / float(q_sum)
        for k in list(quotas.keys()):
            quotas[k] = max(1, int(math.floor(quotas[k] * scale)))

    for station, sdf in df.groupby("station", sort=False):
        k = int(quotas.get(str(station), 0))
        if k <= 0:
            continue
        if sdf.shape[0] <= k:
            parts.append(sdf)
        else:
            parts.append(sdf.sample(n=k, random_state=random_state))

    out = pd.concat(parts, ignore_index=True, sort=False)
    if "timestamp" in out.columns:
        out = out.sort_values(["station", "timestamp"])
    return out


@dataclass(frozen=True)
class XY:
    """Model matrix + target with aligned metadata for evaluation."""

    x: np.ndarray
    y: np.ndarray
    station: np.ndarray
    timestamp: np.ndarray


def to_xy(df: pd.DataFrame, feature_cols: Sequence[str], target: str, drop_any_nan_features: bool) -> XY:
    """Convert DataFrame to model-ready arrays while keeping station/timestamp aligned.

    Leakage note:
      This is purely a transformation for modeling; it does not change the
      chronological ordering.

    Args:
        df: Input DataFrame.
        feature_cols: Feature column list.
        target: Target column name.
        drop_any_nan_features: If True, drop rows where any feature OR target is NaN.
          Use True for sklearn models (Ridge/RF). Use False for XGBoost (it can handle NaNs).

    Returns:
        XY bundle.
    """
    required = [target] + list(feature_cols)
    if drop_any_nan_features:
        # Drop only rows with missing target.
        #
        # IMPORTANT:
        # We intentionally do NOT fill feature NaNs here because sklearn Pipelines
        # already contain a `SimpleImputer(strategy="median")` step. Pre-filling
        # with 0.0 would bias medians, distort scaling, and can hurt performance
        # (especially for stations with sparse sensor coverage).
        clean = df.dropna(subset=[target]).copy()
    else:
        # XGBoost handles NaNs natively
        clean = df.dropna(subset=[target]).copy()

    x = clean.loc[:, list(feature_cols)].to_numpy(dtype=np.float32, copy=False)
    y = clean.loc[:, target].to_numpy(dtype=np.float32, copy=False)
    station = clean["station"].to_numpy(copy=False) if "station" in clean.columns else np.array([])
    timestamp = clean["timestamp"].to_numpy(copy=False) if "timestamp" in clean.columns else np.array([])
    return XY(x=x, y=y, station=station, timestamp=timestamp)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    """Write JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_joblib_bundle(path: Path) -> Dict[str, object]:
    """Load a joblib-saved model bundle.

    This allows resuming training runs (e.g., if XGBoost fails after RF finishes)
    without retraining earlier models.
    """
    import joblib

    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"Unexpected joblib bundle format: {path}")
    return bundle


def tune_ridge_alpha(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target: str,
    alphas: Sequence[float],
) -> Tuple[object, float, Dict[str, float]]:
    """Tune Ridge alpha on validation set and return the best pipeline.

    Hyperparameter rationale (deadline-friendly):
      - `alpha` in {0.1, 1.0, 10.0} provides a quick sweep from light to stronger
        regularization. This stabilizes coefficients under multicollinearity from
        lag/rolling features.
      - `StandardScaler` is included in the pipeline to satisfy normalization
        requirements and to keep the Ridge penalty scale-consistent.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    xy_train = to_xy(train_df, feature_cols, target, drop_any_nan_features=True)
    xy_val = to_xy(val_df, feature_cols, target, drop_any_nan_features=True)

    best_alpha: Optional[float] = None
    best_pipeline: Optional[Pipeline] = None
    best_rmse: Optional[float] = None
    best_metrics: Dict[str, float] = {}

    for a in alphas:
        pipe = Pipeline(
            steps=[
                # Fit-only-on-train imputation avoids dropping large numbers of
                # rows when a minority of sensors are missing in some timestamps.
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("ridge", Ridge(alpha=float(a), fit_intercept=True, random_state=42)),
            ]
        )
        pipe.fit(xy_train.x, xy_train.y)
        pred = pipe.predict(xy_val.x).astype(np.float32)
        m = evaluate(xy_val.y, pred)
        if best_rmse is None or m["rmse"] < best_rmse:
            best_rmse = m["rmse"]
            best_alpha = float(a)
            best_pipeline = pipe
            best_metrics = m

    assert best_alpha is not None and best_pipeline is not None
    return best_pipeline, best_alpha, best_metrics


def train_random_forest(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target: str,
) -> object:
    """Train RandomForestRegressor.

    Uses the project-required defaults:
      - n_estimators=300
      - max_depth=20
      - n_jobs=-1 (falls back to 1 on restricted Windows environments)
      - random_state=42

    Missing values:
      - Uses a median `SimpleImputer` to avoid dropping rows due to sparse sensors.
        The imputer is fit on training data only, so it does not leak information.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    xy_train = to_xy(train_df, feature_cols, target, drop_any_nan_features=True)

    def _fit(n_jobs: int) -> Pipeline:
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=20,
                        n_jobs=n_jobs,
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(xy_train.x, xy_train.y)
        return model

    try:
        return _fit(n_jobs=-1)
    except PermissionError:
        # Some sandboxes deny process/thread primitives used by joblib backend.
        logging.warning("RF parallelism denied (PermissionError). Retrying with n_jobs=1.")
        return _fit(n_jobs=1)


def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target: str,
    params: Optional[Dict[str, object]] = None,
) -> object:
    """Train XGBRegressor with early stopping on validation.

    Hyperparameter rationale (fast + accurate):
      - `n_estimators=1500` with `early_stopping_rounds=50` lets the model stop
        when validation RMSE stalls, avoiding over-training.
      - `learning_rate=0.03` and `max_depth=6` are strong defaults for tabular data.
      - `tree_method="hist"` is fast on CPU.
    """
    try:
        from xgboost import XGBRegressor
    except Exception as e:
        raise SystemExit(f"Missing xgboost. Install requirements_ml.txt. Import error: {e}")

    xy_train = to_xy(train_df, feature_cols, target, drop_any_nan_features=False)
    xy_val = to_xy(val_df, feature_cols, target, drop_any_nan_features=False)

    base_params: Dict[str, object] = {
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "max_depth": 6,
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }
    if params:
        base_params.update(params)

    model = XGBRegressor(**base_params)

    # Compatibility:
    # - Some xgboost builds accept `early_stopping_rounds` in `.fit()`
    # - Others require it in the constructor.
    try:
        model.fit(
            xy_train.x,
            xy_train.y,
            eval_set=[(xy_val.x, xy_val.y)],
            verbose=False,
            early_stopping_rounds=50,
        )
    except TypeError:
        base_params["early_stopping_rounds"] = 50
        model = XGBRegressor(**base_params)
        model.fit(
            xy_train.x,
            xy_train.y,
            eval_set=[(xy_val.x, xy_val.y)],
            verbose=False,
        )
    return model


def tune_xgb(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target: str,
) -> Tuple[object, Dict[str, object], Dict[str, float]]:
    """Tune a small XGBoost parameter set on validation RMSE.

    Rationale (deadline-friendly):
      We use a tiny, high-ROI set of configurations that tends to matter most for
      time-series tabular problems:
        - depth / min_child_weight (controls complexity)
        - reg_lambda (L2 regularization)

    Returns:
        (best_model, best_params, best_val_metrics)
    """
    candidates: List[Dict[str, object]] = [
        {"max_depth": 3, "min_child_weight": 1, "reg_lambda": 1.0},
        {"max_depth": 3, "min_child_weight": 5, "reg_lambda": 1.0},
        {"max_depth": 3, "min_child_weight": 1, "reg_lambda": 5.0},
        {"max_depth": 3, "min_child_weight": 5, "reg_lambda": 5.0},
        {"max_depth": 5, "min_child_weight": 1, "reg_lambda": 1.0},
        {"max_depth": 5, "min_child_weight": 5, "reg_lambda": 1.0},
        {"max_depth": 5, "min_child_weight": 1, "reg_lambda": 5.0},
        {"max_depth": 5, "min_child_weight": 5, "reg_lambda": 5.0},
    ]

    best_model: Optional[object] = None
    best_params: Dict[str, object] = {}
    best_metrics: Dict[str, float] = {}
    best_rmse: Optional[float] = None

    xy_val = to_xy(val_df, feature_cols, target, drop_any_nan_features=False)
    for i, p in enumerate(candidates, start=1):
        logging.info("XGB tune %d/%d: %s", i, len(candidates), p)
        model = train_xgboost(train_df=train_df, val_df=val_df, feature_cols=feature_cols, target=target, params=p)
        pred = np.asarray(model.predict(xy_val.x), dtype=np.float32)
        m = evaluate(xy_val.y, pred)
        if best_rmse is None or m["rmse"] < best_rmse:
            best_rmse = float(m["rmse"])
            best_model = model
            best_params = dict(p)
            best_metrics = m

    assert best_model is not None
    return best_model, best_params, best_metrics


def drop_all_nan_features(df_train: pd.DataFrame, feature_cols: Sequence[str]) -> List[str]:
    """Drop feature columns that are entirely NaN in the given training frame."""
    keep: List[str] = []
    for c in feature_cols:
        if c not in df_train.columns:
            continue
        if not df_train[c].isna().all():
            keep.append(c)
    return keep


def station_model_paths(artifacts: Path, station: str) -> Dict[str, Path]:
    """Return per-station artifact paths."""
    safe = str(station).replace(" ", "_").upper()
    return {
        "ridge": artifacts / f"model_ridge_{safe}.pkl",
        "rf": artifacts / f"model_rf_{safe}.pkl",
        "xgb": artifacts / f"model_xgb_{safe}.json",
        "xgb_meta": artifacts / f"model_xgb_{safe}_meta.json",
    }


def run_per_station(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols_global: Sequence[str],
    target: str,
    artifacts: Path,
    args: argparse.Namespace,
) -> int:
    """Train/evaluate separate models per station and write metrics."""
    import joblib

    stations = sorted([str(s) for s in train_df["station"].dropna().unique()])
    if not stations:
        logging.error("No stations found in training data.")
        return 20

    # Collect overall predictions across all stations for each model.
    y_all: Dict[str, List[np.ndarray]] = {"ridge": [], "rf": [], "xgb": []}
    p_all: Dict[str, List[np.ndarray]] = {"ridge": [], "rf": [], "xgb": []}
    station_all: List[np.ndarray] = []

    by_station: Dict[str, Dict[str, object]] = {}

    for station in stations:
        logging.info("=== Per-station training: %s ===", station)
        tr = train_df[train_df["station"] == station].copy()
        va = val_df[val_df["station"] == station].copy()
        te = test_df[test_df["station"] == station].copy()

        # Drop all-NaN features for this station (imputers can't compute medians).
        feature_cols = drop_all_nan_features(tr, feature_cols_global)
        if not feature_cols:
            logging.warning("Station=%s has no usable feature columns; skipping.", station)
            continue

        # Align columns within this station.
        x_tr, x_va, x_te = align_feature_columns([tr, va, te], feature_cols)
        tr = pd.concat([tr[["station", "timestamp", target]], x_tr], axis=1)
        va = pd.concat([va[["station", "timestamp", target]], x_va], axis=1)
        te = pd.concat([te[["station", "timestamp", target]], x_te], axis=1)

        # Subsampling (already station-filtered, but keep it consistent).
        tr_ridge = subsample_by_station(tr, max_rows=int(getattr(args, "ridge_max_rows", 0)), random_state=42)
        tr_rf = subsample_by_station(tr, max_rows=int(getattr(args, "max_train_rows", 0)), random_state=42)
        tr_xgb = subsample_by_station(tr, max_rows=int(getattr(args, "xgb_max_train_rows", 0)), random_state=42)

        paths = station_model_paths(artifacts, station)

        # Ridge
        ridge_pipe = None
        best_alpha = None
        ridge_val_metrics: Dict[str, float] = {}
        if bool(getattr(args, "skip_ridge", False)):
            bundle = load_joblib_bundle(paths["ridge"])
            ridge_pipe = bundle["model"]
            best_alpha = bundle.get("best_alpha", None)
        else:
            ridge_pipe, best_alpha, ridge_val_metrics = tune_ridge_alpha(
                train_df=tr_ridge,
                val_df=va,
                feature_cols=feature_cols,
                target=target,
                alphas=[0.1, 1.0, 10.0],
            )
            # Fit on train+val
            trva = pd.concat([tr, va], ignore_index=True)
            xy_trva = to_xy(trva, feature_cols, target, drop_any_nan_features=True)
            ridge_pipe.fit(xy_trva.x, xy_trva.y)
            joblib.dump(
                {"model": ridge_pipe, "feature_cols": feature_cols, "target": target, "best_alpha": best_alpha, "ridge_val_metrics": ridge_val_metrics},
                paths["ridge"],
            )

        # RF
        if bool(getattr(args, "skip_rf", False)):
            bundle = load_joblib_bundle(paths["rf"])
            rf_model = bundle["model"]
        else:
            rf_model = train_random_forest(train_df=tr_rf, feature_cols=feature_cols, target=target)
            joblib.dump({"model": rf_model, "feature_cols": feature_cols, "target": target}, paths["rf"])

        # XGB
        xgb_model = None
        xgb_best_params: Dict[str, object] = {}
        xgb_val_metrics: Dict[str, float] = {}
        if bool(getattr(args, "skip_xgb", False)):
            if not paths["xgb"].exists():
                logging.error("Missing XGB artifact for station=%s: %s", station, paths["xgb"])
                return 21
            from xgboost import XGBRegressor

            xgb_model = XGBRegressor()
            xgb_model.load_model(str(paths["xgb"]))
        else:
            if bool(getattr(args, "xgb_tune", False)):
                xgb_model, xgb_best_params, xgb_val_metrics = tune_xgb(
                    train_df=tr_xgb, val_df=va, feature_cols=feature_cols, target=target
                )
            else:
                xgb_model = train_xgboost(train_df=tr_xgb, val_df=va, feature_cols=feature_cols, target=target)
            xgb_model.save_model(str(paths["xgb"]))
            paths["xgb_meta"].write_text(
                json.dumps({"best_params": xgb_best_params, "val_metrics": xgb_val_metrics, "feature_cols": list(feature_cols)}, indent=2),
                encoding="utf-8",
            )

        # Evaluate on station test.
        xy_test_sklearn = to_xy(te, feature_cols, target, drop_any_nan_features=True)
        ridge_pred = ridge_pipe.predict(xy_test_sklearn.x).astype(np.float32)
        rf_pred = np.asarray(rf_model.predict(xy_test_sklearn.x), dtype=np.float32)

        xy_test_xgb = to_xy(te, feature_cols, target, drop_any_nan_features=False)
        xgb_pred = np.asarray(xgb_model.predict(xy_test_xgb.x), dtype=np.float32)

        by_station[station] = {
            "n_features": int(len(feature_cols)),
            "ridge": evaluate(xy_test_sklearn.y, ridge_pred),
            "random_forest": evaluate(xy_test_sklearn.y, rf_pred),
            "xgboost": evaluate(xy_test_xgb.y, xgb_pred),
            "ridge_best_alpha": best_alpha,
            "ridge_val_metrics": ridge_val_metrics,
            "xgb_val_metrics": xgb_val_metrics,
            "xgb_best_params": xgb_best_params,
            "test_rows_sklearn": int(xy_test_sklearn.y.shape[0]),
            "test_rows_xgb": int(xy_test_xgb.y.shape[0]),
        }

        # Collect for overall metrics (note: sklearn eval rows and xgb eval rows match here
        # because we dropped missing target only; feature NaNs are imputed/handled).
        y_all["ridge"].append(xy_test_sklearn.y)
        p_all["ridge"].append(ridge_pred)
        y_all["rf"].append(xy_test_sklearn.y)
        p_all["rf"].append(rf_pred)
        y_all["xgb"].append(xy_test_xgb.y)
        p_all["xgb"].append(xgb_pred)
        station_all.append(np.asarray([station] * int(xy_test_sklearn.y.shape[0])))

    def _cat(xs: List[np.ndarray]) -> np.ndarray:
        return np.concatenate(xs, axis=0) if xs else np.array([], dtype=np.float32)

    overall = {
        "ridge": evaluate(_cat(y_all["ridge"]), _cat(p_all["ridge"])),
        "random_forest": evaluate(_cat(y_all["rf"]), _cat(p_all["rf"])),
        "xgboost": evaluate(_cat(y_all["xgb"]), _cat(p_all["xgb"])),
        "per_station": by_station,
    }
    save_json(artifacts / "classical_metrics_per_station.json", overall)
    logging.info("Saved per-station metrics: %s", artifacts / "classical_metrics_per_station.json")
    return 0


def main() -> int:
    """Train Ridge, Random Forest, and XGBoost models on Phase B feature parquets."""
    parser = argparse.ArgumentParser(description="Train classical models for PM2.5 prediction.")
    parser.add_argument("--train", type=str, default=str(Path("artifacts") / "features_train.parquet"))
    parser.add_argument("--val", type=str, default=str(Path("artifacts") / "features_val.parquet"))
    parser.add_argument("--test", type=str, default=str(Path("artifacts") / "features_test.parquet"))
    parser.add_argument("--target", type=str, default=TARGET_DEFAULT)
    parser.add_argument("--skip-ridge", action="store_true", help="Skip Ridge training and load existing artifact if available.")
    parser.add_argument("--skip-rf", action="store_true", help="Skip Random Forest training and load existing artifact if available.")
    parser.add_argument("--skip-xgb", action="store_true", help="Skip XGBoost training and load existing artifact if available.")
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=50000,
        help="If >0, subsample TRAIN split to this many rows for RF speed; 0 disables.",
    )
    parser.add_argument(
        "--xgb-max-train-rows",
        type=int,
        default=50000,
        help="If >0, subsample TRAIN split to this many rows for XGB speed; 0 disables.",
    )
    parser.add_argument(
        "--ridge-max-rows",
        type=int,
        default=0,
        help="If >0, subsample TRAIN split to this many rows for Ridge tuning; 0 uses full train (default).",
    )
    parser.add_argument(
        "--per-station-metrics",
        action="store_true",
        help="Also include per-station metrics in classical_metrics.json.",
    )
    parser.add_argument(
        "--per-station",
        action="store_true",
        help="Train separate models per station and report overall + per-station metrics.",
    )
    parser.add_argument(
        "--xgb-tune",
        action="store_true",
        help="Run a small, fast XGB parameter sweep on the validation set and fit the best.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    train_df = load_parquet(Path(args.train).resolve())
    val_df = load_parquet(Path(args.val).resolve())
    test_df = load_parquet(Path(args.test).resolve())
    target = str(args.target)

    if target not in train_df.columns:
        logging.error("Target `%s` not found in train parquet.", target)
        return 2

    # Ensure consistent feature columns across splits.
    feature_cols = get_feature_columns(train_df, target=target)
    if not feature_cols:
        logging.error("No numeric feature columns found.")
        return 3

    # Drop all-NaN feature columns based on TRAIN only (imputers cannot compute
    # medians for columns that are completely missing).
    all_nan = train_df[feature_cols].isna().all(axis=0)
    dropped = [c for c, is_all_nan in all_nan.items() if bool(is_all_nan)]
    if dropped:
        logging.info("Dropping %d all-NaN feature columns (train): %s", len(dropped), dropped[:10])
        feature_cols = [c for c in feature_cols if c not in set(dropped)]

    x_train_df, x_val_df, x_test_df = align_feature_columns([train_df, val_df, test_df], feature_cols)
    # Re-attach meta and target for downstream helpers.
    train_df = pd.concat([train_df[["station", "timestamp", target]], x_train_df], axis=1)
    val_df = pd.concat([val_df[["station", "timestamp", target]], x_val_df], axis=1)
    test_df = pd.concat([test_df[["station", "timestamp", target]], x_test_df], axis=1)

    # FAST-TRACK:
    # - Ridge is relatively fast and often benefits from more rows, so by default
    #   we keep the full train set for Ridge (unless capped via --ridge-max-rows).
    # - RF/XGB can be slow on wide, lagged features, so we default to a capped
    #   subsample for them (configurable).
    train_for_ridge = train_df
    if int(args.ridge_max_rows) > 0 and len(train_for_ridge) > int(args.ridge_max_rows):
        logging.info("Sub-sampling train data to %d rows for Ridge...", int(args.ridge_max_rows))
        train_for_ridge = subsample_by_station(train_for_ridge, max_rows=int(args.ridge_max_rows), random_state=42)

    train_for_rf = train_df
    if int(args.max_train_rows) > 0 and len(train_for_rf) > int(args.max_train_rows):
        logging.info("Sub-sampling train data to %d rows for RF speed...", int(args.max_train_rows))
        train_for_rf = subsample_by_station(train_for_rf, max_rows=int(args.max_train_rows), random_state=42)

    train_for_xgb = train_df
    if int(args.xgb_max_train_rows) > 0 and len(train_for_xgb) > int(args.xgb_max_train_rows):
        logging.info("Sub-sampling train data to %d rows for XGB speed...", int(args.xgb_max_train_rows))
        train_for_xgb = subsample_by_station(train_for_xgb, max_rows=int(args.xgb_max_train_rows), random_state=42)

    artifacts = Path("artifacts").resolve()
    artifacts.mkdir(parents=True, exist_ok=True)

    # 1) Ridge (tune alpha on validation, then fit on train+val)
    try:
        import joblib
    except Exception as e:
        logging.error("Missing joblib. Install requirements_ml.txt. err=%s", e)
        return 4

    # Optional: train separate models per station.
    if bool(args.per_station):
        return run_per_station(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_cols_global=feature_cols,
            target=target,
            artifacts=artifacts,
            args=args,
        )

    ridge_path = artifacts / "model_ridge.pkl"
    if args.skip_ridge:
        logging.info("Skipping Ridge training; loading %s", ridge_path)
        ridge_bundle = load_joblib_bundle(ridge_path)
        ridge_pipe = ridge_bundle["model"]
        best_alpha = ridge_bundle.get("best_alpha", None)
        val_metrics = ridge_bundle.get("ridge_val_metrics", {}) if isinstance(ridge_bundle.get("ridge_val_metrics", {}), dict) else {}
    else:
        logging.info("Tuning Ridge alpha on validation...")
        ridge_pipe, best_alpha, val_metrics = tune_ridge_alpha(
            train_df=train_for_ridge,
            val_df=val_df,
            feature_cols=feature_cols,
            target=target,
            alphas=[0.1, 1.0, 10.0],
        )
        logging.info("Best Ridge alpha=%s | val_metrics=%s", best_alpha, val_metrics)

        # Fit best model on Train+Val (per spec)
        trainval_df = pd.concat([train_df, val_df], ignore_index=True)
        xy_trainval = to_xy(trainval_df, feature_cols, target, drop_any_nan_features=True)
        ridge_pipe.fit(xy_trainval.x, xy_trainval.y)

        joblib.dump({"model": ridge_pipe, "feature_cols": feature_cols, "target": target, "best_alpha": best_alpha}, ridge_path)
        logging.info("Saved Ridge model: %s", ridge_path)

    # 2) Random Forest (train on sampled Train)
    rf_path = artifacts / "model_rf.pkl"
    if args.skip_rf:
        logging.info("Skipping RF training; loading %s", rf_path)
        rf_bundle = load_joblib_bundle(rf_path)
        rf_model = rf_bundle["model"]
    else:
        logging.info("Training Random Forest (on sampled data)...")
        rf_model = train_random_forest(train_df=train_for_rf, feature_cols=feature_cols, target=target)
        joblib.dump({"model": rf_model, "feature_cols": feature_cols, "target": target}, rf_path)
        logging.info("Saved RF model: %s", rf_path)

    # 3) XGBoost (train on Train, early stop on Val)
    xgb_path = artifacts / "model_xgb.json"
    xgb_meta_path = artifacts / "model_xgb_meta.json"
    if args.skip_xgb:
        logging.info("Skipping XGB training; expecting existing %s", xgb_path)
        if not xgb_path.exists():
            logging.error("Missing XGB model artifact: %s", xgb_path)
            return 5
        xgb_model = None
        xgb_best_params: Dict[str, object] = {}
        xgb_val_metrics: Dict[str, float] = {}
    else:
        logging.info("Training XGBoost with early stopping...")
        xgb_best_params = {}
        xgb_val_metrics = {}
        if bool(args.xgb_tune):
            xgb_model, xgb_best_params, xgb_val_metrics = tune_xgb(
                train_df=train_for_xgb,
                val_df=val_df,
                feature_cols=feature_cols,
                target=target,
            )
            logging.info("Best XGB params=%s | val_metrics=%s", xgb_best_params, xgb_val_metrics)
        else:
            xgb_model = train_xgboost(train_df=train_for_xgb, val_df=val_df, feature_cols=feature_cols, target=target)
        xgb_model.save_model(xgb_path)
        logging.info("Saved XGB model: %s", xgb_path)
        xgb_meta_path.write_text(
            json.dumps({"best_params": xgb_best_params, "val_metrics": xgb_val_metrics, "feature_cols": list(feature_cols)}, indent=2),
            encoding="utf-8",
        )

    # Quick test evaluation (keeps station/timestamp aligned via to_xy).
    logging.info("Evaluating on test split...")
    xy_test_sklearn = to_xy(test_df, feature_cols, target, drop_any_nan_features=True)
    ridge_pred = ridge_pipe.predict(xy_test_sklearn.x).astype(np.float32)
    rf_pred = np.asarray(rf_model.predict(xy_test_sklearn.x), dtype=np.float32)

    # XGB can use its own NaN handling; evaluate on target-nonnull rows.
    xy_test_xgb = to_xy(test_df, feature_cols, target, drop_any_nan_features=False)
    if xgb_model is None:
        from xgboost import XGBRegressor

        xgb_model = XGBRegressor()
        xgb_model.load_model(xgb_path)
    xgb_pred = np.asarray(xgb_model.predict(xy_test_xgb.x), dtype=np.float32)

    metrics = {
        "ridge": evaluate(xy_test_sklearn.y, ridge_pred),
        "random_forest": evaluate(xy_test_sklearn.y, rf_pred),
        "xgboost": evaluate(xy_test_xgb.y, xgb_pred),
        "ridge_best_alpha": best_alpha,
        "ridge_val_metrics": val_metrics,
        "xgb_val_metrics": xgb_val_metrics if "xgb_val_metrics" in locals() else {},
        "xgb_best_params": xgb_best_params if "xgb_best_params" in locals() else {},
        "n_features": len(feature_cols),
        "test_rows_sklearn": int(xy_test_sklearn.y.shape[0]),
        "test_rows_xgb": int(xy_test_xgb.y.shape[0]),
    }
    if bool(args.per_station_metrics):
        metrics["ridge_by_station"] = evaluate_by_station(xy_test_sklearn.y, ridge_pred, xy_test_sklearn.station)
        metrics["random_forest_by_station"] = evaluate_by_station(xy_test_sklearn.y, rf_pred, xy_test_sklearn.station)
        metrics["xgboost_by_station"] = evaluate_by_station(xy_test_xgb.y, xgb_pred, xy_test_xgb.station)
    save_json(artifacts / "classical_metrics.json", metrics)
    logging.info("Saved metrics: %s", artifacts / "classical_metrics.json")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
