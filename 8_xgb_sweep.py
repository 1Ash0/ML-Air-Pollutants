from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
    """Load a parquet file."""
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


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE/MAE/R2."""
    return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred), "r2": r2(y_true, y_pred)}


def get_feature_columns(df: pd.DataFrame, target: str) -> List[str]:
    """Select numeric feature columns (exclude identifiers and target)."""
    exclude = {"station", "timestamp", "source_file", "source_sheet", target}
    cols: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def align_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align all splits to the same feature columns."""
    train_x = train_df.reindex(columns=list(feature_cols))
    val_x = val_df.reindex(columns=list(feature_cols))
    test_x = test_df.reindex(columns=list(feature_cols))

    train_out = pd.concat([train_df[["station", "timestamp"]], train_df[[c for c in train_df.columns if c == "PM2.5"]], train_x], axis=1)
    val_out = pd.concat([val_df[["station", "timestamp"]], val_df[[c for c in val_df.columns if c == "PM2.5"]], val_x], axis=1)
    test_out = pd.concat([test_df[["station", "timestamp"]], test_df[[c for c in test_df.columns if c == "PM2.5"]], test_x], axis=1)
    return train_out, val_out, test_out


def drop_all_nan_features(train_df: pd.DataFrame, feature_cols: Sequence[str]) -> List[str]:
    """Drop features that are entirely NaN in train (cannot be used by XGBoost)."""
    all_nan = train_df[list(feature_cols)].isna().all(axis=0)
    return [c for c in feature_cols if not bool(all_nan[c])]


def sample_train(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Sample training rows for faster sweep (keeps time order per station afterwards)."""
    if n <= 0 or df.shape[0] <= n:
        return df
    sampled = df.sample(n=n, random_state=42)
    return sampled.sort_values(["station", "timestamp"]).reset_index(drop=True)


def to_matrix(df: pd.DataFrame, feature_cols: Sequence[str], target: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return X/y as float32 arrays, dropping rows with missing target."""
    clean = df.dropna(subset=[target]).copy()
    # Some splits may not contain every feature column (e.g., station-specific sensors).
    # Reindex ensures missing columns are created as NaN instead of raising KeyError.
    x = clean.reindex(columns=list(feature_cols)).to_numpy(dtype=np.float32, copy=False)
    y = clean.loc[:, target].to_numpy(dtype=np.float32, copy=False)
    return x, y


@dataclass(frozen=True)
class SweepConfig:
    """A single XGB config entry."""

    max_depth: int
    min_child_weight: float
    gamma: float

    def name(self) -> str:
        return f"md{self.max_depth}_mcw{self.min_child_weight}_g{self.gamma}"


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Save list of dicts to CSV with stable column ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def save_json(path: Path, payload: Any) -> None:
    """Save JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Automated XGBoost hyperparameter sweep (early stopping on val).")
    parser.add_argument("--train", type=str, default=str(Path("artifacts") / "features_train_v2.parquet"))
    parser.add_argument("--val", type=str, default=str(Path("artifacts") / "features_val_v2.parquet"))
    parser.add_argument("--test", type=str, default=str(Path("artifacts") / "features_test_v2.parquet"))
    parser.add_argument("--target", type=str, default=TARGET_DEFAULT)
    parser.add_argument("--out-csv", type=str, default=str(Path("artifacts") / "xgb_sweep_results.csv"))
    parser.add_argument("--out-json", type=str, default=str(Path("artifacts") / "xgb_sweep_results.json"))
    parser.add_argument("--best-model", type=str, default=str(Path("artifacts") / "model_xgb_best.json"))
    parser.add_argument("--best-meta", type=str, default=str(Path("artifacts") / "model_xgb_best_meta.json"))
    parser.add_argument("--train-sample-rows", type=int, default=200000, help="0 = no sampling (slower).")
    parser.add_argument("--num-boost-round", type=int, default=8000)
    parser.add_argument("--early-stopping-rounds", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--n-jobs", type=int, default=-1)
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
    test_df = load_parquet(Path(args.test).resolve())
    target = str(args.target)

    if target not in train_df.columns:
        logging.error("Target missing in train: %s", target)
        return 2

    feature_cols = get_feature_columns(train_df, target=target)
    if not feature_cols:
        logging.error("No numeric feature columns found.")
        return 3

    feature_cols = drop_all_nan_features(train_df, feature_cols)

    # Sample only training rows for sweep speed; keep full val/test for selection.
    train_df_sweep = sample_train(train_df, int(args.train_sample_rows))

    # Build matrices
    x_train, y_train = to_matrix(train_df_sweep, feature_cols, target)
    x_val, y_val = to_matrix(val_df, feature_cols, target)
    x_test, y_test = to_matrix(test_df, feature_cols, target)

    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=list(feature_cols))
    dval = xgb.DMatrix(x_val, label=y_val, feature_names=list(feature_cols))
    dtest = xgb.DMatrix(x_test, label=y_test, feature_names=list(feature_cols))

    base_params: Dict[str, Any] = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "eta": float(args.learning_rate),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
        "lambda": float(args.reg_lambda),
        "seed": 42,
        "nthread": int(args.n_jobs),
    }

    sweep: List[SweepConfig] = [
        SweepConfig(max_depth=4, min_child_weight=1.0, gamma=0.0),
        SweepConfig(max_depth=4, min_child_weight=5.0, gamma=0.0),
        SweepConfig(max_depth=6, min_child_weight=1.0, gamma=0.0),
        SweepConfig(max_depth=6, min_child_weight=5.0, gamma=0.0),
        SweepConfig(max_depth=4, min_child_weight=1.0, gamma=1.0),
        SweepConfig(max_depth=4, min_child_weight=5.0, gamma=1.0),
        SweepConfig(max_depth=6, min_child_weight=1.0, gamma=1.0),
        SweepConfig(max_depth=6, min_child_weight=5.0, gamma=1.0),
    ]

    rows: List[Dict[str, Any]] = []
    best = None
    best_row: Dict[str, Any] | None = None

    logging.info(
        "Starting sweep: %d configs | train=%s val=%s test=%s | features=%d",
        len(sweep),
        x_train.shape,
        x_val.shape,
        x_test.shape,
        len(feature_cols),
    )

    for i, cfg in enumerate(sweep, start=1):
        params = dict(base_params)
        params.update(
            {
                "max_depth": int(cfg.max_depth),
                "min_child_weight": float(cfg.min_child_weight),
                "gamma": float(cfg.gamma),
            }
        )

        logging.info("(%d/%d) Training %s ...", i, len(sweep), cfg.name())
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=int(args.num_boost_round),
            evals=[(dval, "val")],
            early_stopping_rounds=int(args.early_stopping_rounds),
            verbose_eval=False,
        )

        # Evaluate
        pred_val = booster.predict(dval)
        pred_test = booster.predict(dtest)

        m_val = evaluate(y_val, pred_val)
        m_test = evaluate(y_test, pred_test)

        row = {
            "config": cfg.name(),
            "max_depth": cfg.max_depth,
            "min_child_weight": cfg.min_child_weight,
            "gamma": cfg.gamma,
            "best_iteration": int(getattr(booster, "best_iteration", -1)),
            "val_rmse": m_val["rmse"],
            "val_mae": m_val["mae"],
            "val_r2": m_val["r2"],
            "test_rmse": m_test["rmse"],
            "test_mae": m_test["mae"],
            "test_r2": m_test["r2"],
        }
        rows.append(row)

        # Choose best by validation RMSE, tie-break by validation R².
        if best_row is None:
            best_row = row
            best = booster
        else:
            better = (row["val_rmse"] < best_row["val_rmse"]) or (
                math.isclose(row["val_rmse"], best_row["val_rmse"]) and row["val_r2"] > best_row["val_r2"]
            )
            if bool(better):
                best_row = row
                best = booster

        logging.info("    val_rmse=%.4f val_r2=%.4f | test_rmse=%.4f test_r2=%.4f", row["val_rmse"], row["val_r2"], row["test_rmse"], row["test_r2"])

    # Sort for reporting
    rows_sorted = sorted(rows, key=lambda r: (r["val_rmse"], -r["val_r2"]))

    out_csv = Path(args.out_csv).resolve()
    out_json = Path(args.out_json).resolve()
    save_csv(out_csv, rows_sorted)
    save_json(out_json, {"results": rows_sorted, "best": best_row})

    if best is not None and best_row is not None:
        best_model_path = Path(args.best_model).resolve()
        best_meta_path = Path(args.best_meta).resolve()
        best.save_model(best_model_path)
        save_json(
            best_meta_path,
            {
                "best": best_row,
                "base_params": base_params,
                "train_sample_rows": int(args.train_sample_rows),
                "num_boost_round": int(args.num_boost_round),
                "early_stopping_rounds": int(args.early_stopping_rounds),
                "feature_cols": list(feature_cols),
            },
        )
        logging.info("Best config: %s | saved model=%s", best_row["config"], best_model_path)

    logging.info("Sweep complete. Results: %s", out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
