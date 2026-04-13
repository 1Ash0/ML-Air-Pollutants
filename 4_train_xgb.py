from __future__ import annotations

import argparse
import json
import logging
import sys
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
    """Select numeric feature columns."""
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


def save_json(path: Path, payload: Dict[str, object]) -> None:
    """Save payload as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    """Train an XGBoost regressor with early stopping."""
    parser = argparse.ArgumentParser(description="Train XGBoost on engineered features.")
    parser.add_argument("--train", type=str, default=str(Path("artifacts") / "features_train.parquet"))
    parser.add_argument("--val", type=str, default=str(Path("artifacts") / "features_val.parquet"))
    parser.add_argument("--out-dir", type=str, default=str(Path("artifacts") / "models"))
    parser.add_argument("--target", type=str, default=TARGET_COL_DEFAULT)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        import xgboost as xgb
    except Exception as e:
        logging.error("Missing xgboost. Install requirements_ml.txt. err=%s", e)
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

    x_train, y_train = split_xy(train_df, feature_cols, target_col)
    x_val, y_val = split_xy(val_df, feature_cols, target_col)

    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=list(feature_cols))
    dval = xgb.DMatrix(x_val, label=y_val, feature_names=list(feature_cols))

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "seed": 42,
    }

    logging.info("Training XGBoost with early stopping...")
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1500,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=50,
    )

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "xgb.json"
    meta_path = out_dir / "xgb_meta.json"

    booster.save_model(model_path)
    save_json(
        meta_path,
        {
            "target": target_col,
            "feature_cols": list(feature_cols),
            "best_iteration": int(getattr(booster, "best_iteration", -1)),
            "best_score": float(getattr(booster, "best_score", float("nan"))),
            "params": params,
        },
    )

    logging.info("Saved XGB model to %s", model_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise

