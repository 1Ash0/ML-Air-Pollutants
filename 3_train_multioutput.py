from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


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
    """Load parquet into a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet: {path}")
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
    """R² score (1 - SSE/SST)."""
    denom = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if denom == 0.0:
        return float("nan")
    num = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - (num / denom)


def evaluate_multi(y_true: np.ndarray, y_pred: np.ndarray, target_names: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Compute per-target metrics for multi-output regression."""
    out: Dict[str, Dict[str, float]] = {}
    for i, t in enumerate(target_names):
        yt = np.asarray(y_true[:, i], dtype=np.float32)
        yp = np.asarray(y_pred[:, i], dtype=np.float32)
        out[str(t)] = {"rmse": metric_rmse(yt, yp), "mae": metric_mae(yt, yp), "r2": metric_r2(yt, yp)}
    return out


def mean_rmse(metrics: Dict[str, Dict[str, float]]) -> float:
    """Average RMSE across targets (used for quick validation tuning)."""
    rmses = [float(v["rmse"]) for v in metrics.values() if isinstance(v, dict) and "rmse" in v]
    return float(np.mean(rmses)) if rmses else float("inf")


def available_default_targets(df: pd.DataFrame) -> List[str]:
    """Choose a sensible default set of pollutant targets present in the dataset.

    These are *raw* (non-lag/rolling) columns; engineered variants are used as features.
    """
    preferred = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3", "NO", "NOX", "BENZ"]
    present = [c for c in preferred if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    # Keep at least PM2.5 if present; otherwise raise.
    if not present:
        raise ValueError("No default pollutant targets found in input dataframe.")
    return present


def get_feature_columns(df: pd.DataFrame, exclude_cols: Sequence[str]) -> List[str]:
    """Select numeric feature columns, excluding identifiers and target columns."""
    exclude = set(exclude_cols) | set(META_COLS)
    cols: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and not df[c].isna().all():
            cols.append(str(c))
    return cols


def subsample_by_station(df: pd.DataFrame, max_rows: int, seed: int = 42) -> pd.DataFrame:
    """Station-preserving subsample for speed."""
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    if "station" not in df.columns:
        return df.sample(n=max_rows, random_state=seed)
    parts: List[pd.DataFrame] = []
    stations = df["station"].astype(str)
    rng = np.random.default_rng(seed)
    for st in sorted(stations.unique()):
        sdf = df.loc[stations == st]
        frac = max_rows / max(len(df), 1)
        n = int(max(1, round(frac * len(sdf))))
        n = min(n, len(sdf))
        # Use numpy RNG for reproducibility across pandas versions.
        idx = rng.choice(sdf.index.to_numpy(), size=n, replace=False)
        parts.append(sdf.loc[idx])
    out = pd.concat(parts, axis=0, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def add_horizon_targets(
    df: pd.DataFrame,
    targets: Sequence[str],
    horizon_steps: int,
) -> pd.DataFrame:
    """Create horizon targets (t + horizon_steps) per station.

    We do *forecasting* instead of nowcasting:
      X at time t predicts pollutant concentrations at t+h (default h=1 => +15 minutes).

    This is a clean way to satisfy the "predict multiple pollutants" requirement without
    leaking the current value of a pollutant into its own target.
    """
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be >= 1 for forecasting targets.")
    out = df.copy()
    if "station" not in out.columns:
        raise ValueError("Expected a 'station' column for station-safe shifting.")
    g = out.groupby("station", sort=False)
    for t in targets:
        out[f"y_{t}_t+{horizon_steps}"] = g[t].shift(-horizon_steps)
    return out


def drop_rows_with_missing_targets(df: pd.DataFrame, y_cols: Sequence[str]) -> pd.DataFrame:
    """Drop rows with missing y values (cannot train/evaluate without labels)."""
    clean = df.dropna(subset=list(y_cols)).copy()
    return clean


def build_xy(df: pd.DataFrame, feature_cols: Sequence[str], y_cols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, Y) matrices."""
    x = df.reindex(columns=list(feature_cols)).to_numpy(dtype=np.float32, copy=False)
    y = df.reindex(columns=list(y_cols)).to_numpy(dtype=np.float32, copy=False)
    return x, y


@dataclass(frozen=True)
class ModelBundle:
    """Saved bundle format for multi-output models."""

    model: object
    feature_cols: List[str]
    target_cols: List[str]
    horizon_steps: int


def save_joblib_bundle(path: Path, bundle: ModelBundle) -> None:
    """Save a model bundle with metadata."""
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": bundle.model,
        "feature_cols": bundle.feature_cols,
        "target_cols": bundle.target_cols,
        "horizon_steps": int(bundle.horizon_steps),
    }
    joblib.dump(payload, path)


def train_multi_ridge(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: Sequence[str],
    y_cols: Sequence[str],
    alphas: Sequence[float],
) -> Tuple[object, float, Dict[str, Dict[str, float]]]:
    """Train a multi-target Ridge regression with simple alpha tuning on validation.

    Architecture:
      SimpleImputer(median) -> StandardScaler -> Ridge

    Notes:
      - Ridge naturally supports multi-output targets (y is 2D).
      - Scaling is critical: different engineered features can differ by orders of magnitude.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x_train, y_train = build_xy(train_df, feature_cols, y_cols)
    x_val, y_val = build_xy(val_df, feature_cols, y_cols)

    best_alpha: Optional[float] = None
    best_model: Optional[object] = None
    best_metrics: Dict[str, Dict[str, float]] = {}
    best_score = float("inf")

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
        m = evaluate_multi(y_val, pred, target_names=y_cols)
        score = mean_rmse(m)
        logging.info("Ridge alpha=%s | val_mean_rmse=%.4f", a, score)
        if score < best_score:
            best_score = score
            best_alpha = float(a)
            best_model = model
            best_metrics = m

    if best_model is None or best_alpha is None:
        raise RuntimeError("Failed to train Ridge.")
    return best_model, best_alpha, best_metrics


def train_multi_rf(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    y_cols: Sequence[str],
    n_estimators: int,
    max_depth: int,
) -> object:
    """Train a multi-output RandomForestRegressor.

    RandomForestRegressor natively supports multi-output regression when y is 2D.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    x_train, y_train = build_xy(train_df, feature_cols, y_cols)
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=int(n_estimators),
                    max_depth=int(max_depth),
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model


def train_multi_xgb(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    y_cols: Sequence[str],
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
) -> object:
    """Train a multi-output XGBoost regressor via MultiOutputRegressor.

    We avoid early-stopping here to keep the multi-target pipeline simple and robust:
    each target is an independent XGBoost model wrapped under a single estimator.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.pipeline import Pipeline
    from xgboost import XGBRegressor

    x_train, y_train = build_xy(train_df, feature_cols, y_cols)
    base = XGBRegressor(
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("xgb", MultiOutputRegressor(base))])
    model.fit(x_train, y_train)
    return model


def evaluate_and_write_csv(
    test_df: pd.DataFrame,
    model: object,
    feature_cols: Sequence[str],
    y_cols: Sequence[str],
    out_csv: Path,
    model_name: str,
) -> None:
    """Evaluate on test and write per-target metrics (overall + per station) to CSV."""
    x_test, y_test = build_xy(test_df, feature_cols, y_cols)
    pred = np.asarray(model.predict(x_test), dtype=np.float32)
    metrics_overall = evaluate_multi(y_test, pred, target_names=y_cols)

    rows: List[Dict[str, object]] = []
    for t, m in metrics_overall.items():
        rows.append({"station": "ALL", "model": model_name, "target": t, **m, "n_rows": int(len(test_df))})

    if "station" in test_df.columns:
        stations = test_df["station"].astype(str).to_numpy()
        for st in np.unique(stations):
            mask = stations == st
            if int(np.sum(mask)) < 50:
                continue
            m_st = evaluate_multi(y_test[mask], pred[mask], target_names=y_cols)
            for t, m in m_st.items():
                rows.append({"station": st, "model": model_name, "target": t, **m, "n_rows": int(np.sum(mask))})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    if out_csv.exists():
        # Append (so you can run Ridge then RF without overwriting).
        df_prev = pd.read_csv(out_csv)
        df_out = pd.concat([df_prev, df_out], ignore_index=True)
    df_out.to_csv(out_csv, index=False)
    logging.info("Wrote multi-output metrics: %s", out_csv)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train multi-pollutant (multi-output) models with station-safe forecasting targets."
    )
    parser.add_argument("--train", type=str, default=str(Path("artifacts") / "features_train_v4.parquet"))
    parser.add_argument("--val", type=str, default=str(Path("artifacts") / "features_val_v4.parquet"))
    parser.add_argument("--test", type=str, default=str(Path("artifacts") / "features_test_v4.parquet"))
    parser.add_argument("--targets", type=str, default="", help="Comma-separated pollutant targets; default autodetect.")
    parser.add_argument("--horizon-steps", type=int, default=1, help="Forecast horizon in 15-min steps (default 1).")
    parser.add_argument("--max-train-rows", type=int, default=0, help="Optional station-preserving subsample cap.")
    parser.add_argument("--artifacts", type=str, default=str(Path("artifacts")))
    parser.add_argument("--out-metrics-csv", type=str, default=str(Path("artifacts") / "multioutput_metrics.csv"))
    parser.add_argument("--save-models", action="store_true", help="Persist trained models to artifacts/ for submission.")
    parser.add_argument("--skip-rf", action="store_true")
    parser.add_argument("--skip-xgb", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    artifacts = Path(args.artifacts)
    out_csv = Path(args.out_metrics_csv)

    train_path = resolve_input_path(Path(args.train), "features_train_v4.parquet")
    val_path = resolve_input_path(Path(args.val), "features_val_v4.parquet")
    test_path = resolve_input_path(Path(args.test), "features_test_v4.parquet")

    logging.info("Loading train/val/test...")
    train_df = load_parquet(train_path)
    val_df = load_parquet(val_path)
    test_df = load_parquet(test_path)

    # Targets (raw pollutant columns)
    targets = [t.strip() for t in args.targets.split(",") if t.strip()] if args.targets else available_default_targets(train_df)
    horizon = int(args.horizon_steps)
    logging.info("Targets=%s | horizon_steps=%s (+%d minutes)", targets, horizon, horizon * 15)

    # Create station-safe forecast targets (t+h). Drop rows without labels.
    train_df = add_horizon_targets(train_df, targets=targets, horizon_steps=horizon)
    val_df = add_horizon_targets(val_df, targets=targets, horizon_steps=horizon)
    test_df = add_horizon_targets(test_df, targets=targets, horizon_steps=horizon)

    y_cols = [f"y_{t}_t+{horizon}" for t in targets]
    train_df = drop_rows_with_missing_targets(train_df, y_cols=y_cols)
    val_df = drop_rows_with_missing_targets(val_df, y_cols=y_cols)
    test_df = drop_rows_with_missing_targets(test_df, y_cols=y_cols)

    # Optional speed subsample (train only)
    if int(args.max_train_rows) > 0:
        train_df = subsample_by_station(train_df, max_rows=int(args.max_train_rows), seed=42)
        logging.info("Subsampled train rows => %d", len(train_df))

    # Use the same feature set across splits.
    feature_cols = get_feature_columns(train_df, exclude_cols=list(targets) + list(y_cols))
    logging.info("n_features=%d | n_targets=%d", len(feature_cols), len(y_cols))

    # Align columns across splits (parquet schema is stable, but we reindex defensively).
    train_df = train_df.reindex(columns=list(META_COLS) + list(targets) + list(y_cols) + feature_cols)
    val_df = val_df.reindex(columns=list(META_COLS) + list(targets) + list(y_cols) + feature_cols)
    test_df = test_df.reindex(columns=list(META_COLS) + list(targets) + list(y_cols) + feature_cols)

    # 1) Ridge (tuned)
    logging.info("Training multi-output Ridge (alpha tuning on validation)...")
    alphas = [0.1, 1.0, 10.0]
    ridge_model, best_alpha, ridge_val_metrics = train_multi_ridge(
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        y_cols=y_cols,
        alphas=alphas,
    )
    logging.info("Best Ridge alpha=%s | val_mean_rmse=%.4f", best_alpha, mean_rmse(ridge_val_metrics))
    evaluate_and_write_csv(
        test_df=test_df,
        model=ridge_model,
        feature_cols=feature_cols,
        y_cols=y_cols,
        out_csv=out_csv,
        model_name="multi_ridge",
    )

    if args.save_models:
        save_joblib_bundle(
            artifacts / "model_multi_ridge.pkl",
            ModelBundle(
                model=ridge_model,
                feature_cols=list(feature_cols),
                target_cols=list(y_cols),
                horizon_steps=horizon,
            ),
        )
        (artifacts / "multioutput_ridge_meta.json").write_text(
            json.dumps(
                {
                    "raw_targets": targets,
                    "target_cols": y_cols,
                    "horizon_steps": horizon,
                    "best_alpha": best_alpha,
                    "val_metrics": ridge_val_metrics,
                    "n_features": len(feature_cols),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # 2) Random Forest
    if not args.skip_rf:
        logging.info("Training multi-output RandomForestRegressor...")
        rf_model = train_multi_rf(
            train_df=train_df,
            feature_cols=feature_cols,
            y_cols=y_cols,
            n_estimators=300,
            max_depth=20,
        )
        evaluate_and_write_csv(
            test_df=test_df,
            model=rf_model,
            feature_cols=feature_cols,
            y_cols=y_cols,
            out_csv=out_csv,
            model_name="multi_rf",
        )
        if args.save_models:
            save_joblib_bundle(
                artifacts / "model_multi_rf.pkl",
                ModelBundle(
                    model=rf_model,
                    feature_cols=list(feature_cols),
                    target_cols=list(y_cols),
                    horizon_steps=horizon,
                ),
            )

    # 3) XGBoost (optional)
    if not args.skip_xgb:
        logging.info("Training multi-output XGBoost (MultiOutputRegressor)...")
        xgb_model = train_multi_xgb(
            train_df=train_df,
            feature_cols=feature_cols,
            y_cols=y_cols,
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
        )
        evaluate_and_write_csv(
            test_df=test_df,
            model=xgb_model,
            feature_cols=feature_cols,
            y_cols=y_cols,
            out_csv=out_csv,
            model_name="multi_xgb",
        )
        if args.save_models:
            save_joblib_bundle(
                artifacts / "model_multi_xgb.pkl",
                ModelBundle(
                    model=xgb_model,
                    feature_cols=list(feature_cols),
                    target_cols=list(y_cols),
                    horizon_steps=horizon,
                ),
            )

    # Final metadata for viva write-up.
    (artifacts / "multioutput_meta.json").write_text(
        json.dumps(
            {
                "raw_targets": targets,
                "target_cols": y_cols,
                "horizon_steps": horizon,
                "n_features": len(feature_cols),
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "test_rows": int(len(test_df)),
                "notes": [
                    "Targets are forecasting labels (t + horizon_steps) computed per station.",
                    "Features include engineered lags/rolling means at time t; no station-crossing sequences.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logging.info("Wrote: %s", artifacts / "multioutput_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

