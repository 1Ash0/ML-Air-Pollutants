from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file."""
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
    """Compute R²."""
    denom = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if denom == 0.0:
        return float("nan")
    num = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - (num / denom)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression metrics."""
    return {"rmse": metric_rmse(y_true, y_pred), "mae": metric_mae(y_true, y_pred), "r2": metric_r2(y_true, y_pred)}


def station_model_paths(artifacts: Path, station: str) -> Dict[str, Path]:
    """Return per-station artifact paths produced by `3_train_classical.py --per-station`."""
    safe = str(station).replace(" ", "_").upper()
    return {
        "ridge": artifacts / f"model_ridge_{safe}.pkl",
        "rf": artifacts / f"model_rf_{safe}.pkl",
        "xgb": artifacts / f"model_xgb_{safe}.json",
        "xgb_meta": artifacts / f"model_xgb_{safe}_meta.json",
    }


@dataclass(frozen=True)
class ModelBundle:
    """A loaded model + its expected feature columns."""

    kind: str  # "ridge" | "rf" | "xgb"
    model: object
    feature_cols: List[str]


def load_ridge_or_rf(path: Path) -> ModelBundle:
    """Load a joblib bundle saved by `3_train_classical.py`."""
    import joblib

    if not path.exists():
        raise FileNotFoundError(path)
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"Unexpected joblib format: {path}")
    feature_cols = bundle.get("feature_cols", [])
    if not isinstance(feature_cols, list) or not feature_cols:
        raise ValueError(f"Missing feature_cols in {path}")
    kind = "ridge" if "ridge" in path.name.lower() else "rf"
    return ModelBundle(kind=kind, model=bundle["model"], feature_cols=[str(c) for c in feature_cols])


def load_xgb(model_path: Path, meta_path: Path) -> ModelBundle:
    """Load an XGBoost model + feature columns meta saved by `3_train_classical.py`."""
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_cols = meta.get("feature_cols", [])
    if not isinstance(feature_cols, list) or not feature_cols:
        raise ValueError(f"Missing feature_cols in {meta_path}")

    from xgboost import XGBRegressor

    model = XGBRegressor()
    model.load_model(str(model_path))
    return ModelBundle(kind="xgb", model=model, feature_cols=[str(c) for c in feature_cols])


def build_xy(df: pd.DataFrame, feature_cols: Sequence[str], target: str) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, y) with target-only row filtering and column reindexing."""
    clean = df.dropna(subset=[target]).copy()
    x = clean.reindex(columns=list(feature_cols)).to_numpy(dtype=np.float32, copy=False)
    y = clean[target].to_numpy(dtype=np.float32, copy=False)
    return x, y


def predict(bundle: ModelBundle, df: pd.DataFrame, target: str) -> Tuple[np.ndarray, np.ndarray]:
    """Predict for a split and return (y_true, y_pred)."""
    x, y = build_xy(df, bundle.feature_cols, target)
    pred = np.asarray(bundle.model.predict(x), dtype=np.float32)
    return y, pred


def choose_best_model_for_station(
    candidates: List[ModelBundle],
    val_df_station: pd.DataFrame,
    target: str,
) -> Tuple[ModelBundle, Dict[str, Dict[str, float]]]:
    """Choose the best model by validation RMSE for a station."""
    scores: Dict[str, Dict[str, float]] = {}
    best: Optional[ModelBundle] = None
    best_rmse: Optional[float] = None
    for b in candidates:
        y, p = predict(b, val_df_station, target)
        m = evaluate(y, p)
        scores[b.kind] = m
        if best_rmse is None or m["rmse"] < best_rmse:
            best_rmse = float(m["rmse"])
            best = b
    if best is None:
        raise RuntimeError("No valid candidate models to choose from.")
    return best, scores


def main() -> int:
    parser = argparse.ArgumentParser(description="Route best model per station based on validation RMSE.")
    parser.add_argument("--train", type=str, default=str(Path("artifacts") / "features_train_v3.parquet"))
    parser.add_argument("--val", type=str, default=str(Path("artifacts") / "features_val_v3.parquet"))
    parser.add_argument("--test", type=str, default=str(Path("artifacts") / "features_test_v3.parquet"))
    parser.add_argument("--target", type=str, default=TARGET_DEFAULT)
    parser.add_argument("--artifacts", type=str, default=str(Path("artifacts")))
    parser.add_argument("--out-json", type=str, default=str(Path("artifacts") / "classical_metrics_routed.json"))
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    target = str(args.target)
    artifacts = Path(args.artifacts).resolve()

    train_df = load_parquet(Path(args.train).resolve())
    val_df = load_parquet(Path(args.val).resolve())
    test_df = load_parquet(Path(args.test).resolve())

    if "station" not in train_df.columns or target not in train_df.columns:
        logging.error("Expected `station` and target column `%s` in inputs.", target)
        return 2

    stations = sorted([str(s) for s in train_df["station"].dropna().unique()])
    logging.info("Stations=%s", stations)

    routed_by_station: Dict[str, Dict[str, Any]] = {}
    y_all: List[np.ndarray] = []
    p_all: List[np.ndarray] = []
    station_all: List[np.ndarray] = []

    for station in stations:
        paths = station_model_paths(artifacts, station)
        candidates: List[ModelBundle] = []

        # Load available station models (skip missing).
        try:
            candidates.append(load_ridge_or_rf(paths["ridge"]))
        except Exception as e:
            logging.warning("Station=%s | ridge missing/unreadable: %s", station, e)
        try:
            candidates.append(load_ridge_or_rf(paths["rf"]))
        except Exception as e:
            logging.warning("Station=%s | rf missing/unreadable: %s", station, e)
        try:
            candidates.append(load_xgb(paths["xgb"], paths["xgb_meta"]))
        except Exception as e:
            logging.warning("Station=%s | xgb missing/unreadable: %s", station, e)

        if not candidates:
            logging.error("No per-station models found for station=%s. Run `3_train_classical.py --per-station` first.", station)
            return 3

        val_station = val_df[val_df["station"] == station].copy()
        test_station = test_df[test_df["station"] == station].copy()

        best, val_scores = choose_best_model_for_station(candidates, val_station, target=target)
        y_test, p_test = predict(best, test_station, target=target)
        test_metrics = evaluate(y_test, p_test)

        routed_by_station[station] = {
            "chosen_model": best.kind,
            "val_scores": val_scores,
            "test_metrics": test_metrics,
            "n_test_rows": int(y_test.shape[0]),
            "n_features": int(len(best.feature_cols)),
        }

        y_all.append(y_test)
        p_all.append(p_test)
        station_all.append(np.asarray([station] * int(y_test.shape[0])))

        logging.info("Station=%s | chosen=%s | test=%s", station, best.kind, test_metrics)

    y_concat = np.concatenate(y_all, axis=0) if y_all else np.array([], dtype=np.float32)
    p_concat = np.concatenate(p_all, axis=0) if p_all else np.array([], dtype=np.float32)
    overall = evaluate(y_concat, p_concat) if y_all else {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

    payload = {
        "overall": overall,
        "target": target,
        "station_routing": routed_by_station,
    }

    out_p = Path(args.out_json).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Wrote: %s", out_p)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise

