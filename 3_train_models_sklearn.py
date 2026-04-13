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


TARGET_COL_DEFAULT: str = "PM2.5"


def setup_logging(level: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_parquet(path: Path) -> pd.DataFrame:
    """Load parquet into a DataFrame.

    Args:
        path: Parquet path.

    Returns:
        DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet: {path}")
    return pd.read_parquet(path)


def select_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    """Select model feature columns (exclude identifiers and target).

    Args:
        df: Input DataFrame.
        target_col: Target column name.

    Returns:
        List of feature column names.
    """
    exclude = {"station", "timestamp", "source_file", "source_sheet", target_col}
    cols: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def split_xy(df: pd.DataFrame, feature_cols: Sequence[str], target_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """Create X/y arrays for sklearn.

    Args:
        df: Input DataFrame.
        feature_cols: Feature column list.
        target_col: Target column name.

    Returns:
        (X, y) as numpy arrays.
    """
    x = df.loc[:, list(feature_cols)].to_numpy(dtype=np.float32, copy=False)
    y = df.loc[:, target_col].to_numpy(dtype=np.float32, copy=False)
    return x, y


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


@dataclass(frozen=True)
class ModelResult:
    """Container for quick validation metrics."""

    model_name: str
    params: Dict[str, object]
    rmse: float
    mae: float


def train_ridge(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    alphas: Sequence[float],
) -> Tuple[object, ModelResult]:
    """Train a Ridge regression model (fast baseline).

    Note: Features are already standardized in Phase B output, but Ridge remains
    useful for its stability and speed.
    """
    from sklearn.linear_model import Ridge

    x_train, y_train = split_xy(train_df, feature_cols, target_col)
    x_val, y_val = split_xy(val_df, feature_cols, target_col)

    best_model: Optional[Ridge] = None
    best_res: Optional[ModelResult] = None
    for a in alphas:
        model = Ridge(alpha=float(a), fit_intercept=True, random_state=42)
        model.fit(x_train, y_train)
        pred = model.predict(x_val).astype(np.float32)
        res = ModelResult(
            model_name="ridge",
            params={"alpha": float(a)},
            rmse=rmse(y_val, pred),
            mae=mae(y_val, pred),
        )
        if best_res is None or res.rmse < best_res.rmse:
            best_model = model
            best_res = res

    assert best_model is not None and best_res is not None
    return best_model, best_res


def train_random_forest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    max_depth_options: Sequence[Optional[int]],
) -> Tuple[object, ModelResult]:
    """Train a RandomForestRegressor with minimal tuning."""
    from sklearn.ensemble import RandomForestRegressor

    x_train, y_train = split_xy(train_df, feature_cols, target_col)
    x_val, y_val = split_xy(val_df, feature_cols, target_col)

    best_model: Optional[RandomForestRegressor] = None
    best_res: Optional[ModelResult] = None

    for depth in max_depth_options:
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=depth,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_val).astype(np.float32)
        res = ModelResult(
            model_name="random_forest",
            params={"max_depth": depth},
            rmse=rmse(y_val, pred),
            mae=mae(y_val, pred),
        )
        if best_res is None or res.rmse < best_res.rmse:
            best_model = model
            best_res = res

    assert best_model is not None and best_res is not None
    return best_model, best_res


def save_json(path: Path, payload: Dict[str, object]) -> None:
    """Save payload as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    """Train Ridge + RandomForest models and persist artifacts."""
    parser = argparse.ArgumentParser(description="Train sklearn baselines on engineered features.")
    parser.add_argument("--train", type=str, default=str(Path("artifacts") / "features_train.parquet"))
    parser.add_argument("--val", type=str, default=str(Path("artifacts") / "features_val.parquet"))
    parser.add_argument("--out-dir", type=str, default=str(Path("artifacts") / "models"))
    parser.add_argument("--target", type=str, default=TARGET_COL_DEFAULT)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    train_path = Path(args.train).resolve()
    val_path = Path(args.val).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    target_col = str(args.target)

    logging.info("Loading train: %s", train_path)
    train_df = load_parquet(train_path)
    logging.info("Loading val: %s", val_path)
    val_df = load_parquet(val_path)

    if target_col not in train_df.columns:
        logging.error("Target column missing in train: %s", target_col)
        return 2

    feature_cols = select_feature_columns(train_df, target_col=target_col)
    if not feature_cols:
        logging.error("No numeric feature columns found.")
        return 3

    # Guard: sklearn models cannot handle NaNs.
    req_cols = [target_col] + feature_cols
    train_df = train_df.dropna(subset=req_cols).reset_index(drop=True)
    val_df = val_df.dropna(subset=req_cols).reset_index(drop=True)

    logging.info("Training Ridge...")
    ridge_model, ridge_res = train_ridge(
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        target_col=target_col,
        alphas=[0.1, 1.0, 10.0],
    )
    logging.info("Ridge best: %s", ridge_res)

    logging.info("Training RandomForest...")
    rf_model, rf_res = train_random_forest(
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        target_col=target_col,
        max_depth_options=[10, 20, None],
    )
    logging.info("RF best: %s", rf_res)

    # Persist models.
    try:
        import joblib
    except Exception as e:
        logging.error("Missing joblib. Install requirements_ml.txt. err=%s", e)
        return 4

    joblib.dump(
        {"model": ridge_model, "feature_cols": list(feature_cols), "target": target_col},
        out_dir / "ridge.joblib",
    )
    joblib.dump(
        {"model": rf_model, "feature_cols": list(feature_cols), "target": target_col},
        out_dir / "random_forest.joblib",
    )

    # Persist quick val metrics for later reporting.
    save_json(
        out_dir / "sklearn_val_metrics.json",
        {
            "ridge": {"params": ridge_res.params, "rmse": ridge_res.rmse, "mae": ridge_res.mae},
            "random_forest": {"params": rf_res.params, "rmse": rf_res.rmse, "mae": rf_res.mae},
        },
    )

    logging.info("Saved models to %s", out_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise

