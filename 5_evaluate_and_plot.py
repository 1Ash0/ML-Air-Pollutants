from __future__ import annotations

import argparse
import json
import logging
import math
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
    """Load parquet into a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet: {path}")
    return pd.read_parquet(path)


def metric_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def metric_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE."""
    return float(np.mean(np.abs(y_true - y_pred)))


def metric_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² score."""
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


def ensure_plot_style() -> None:
    """Configure seaborn/matplotlib for a clean academic look."""
    import matplotlib as mpl
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")
    sns.set_palette("deep")

    mpl.rcParams.update(
        {
            "axes.facecolor": "#FBFBFB",
            "figure.facecolor": "white",
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.8,
            "axes.edgecolor": "#2B2B2B",
            "axes.linewidth": 1.0,
            "axes.titleweight": "bold",
            "font.size": 12,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


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
class SklearnBundle:
    """A joblib-saved sklearn model bundle with explicit feature columns."""

    model: object
    feature_cols: List[str]
    target: str


def load_joblib_bundle(path: Path) -> SklearnBundle:
    """Load Ridge/RF bundle saved by `3_train_classical.py`."""
    import joblib

    if not path.exists():
        raise FileNotFoundError(path)
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"Unexpected joblib bundle format: {path}")
    feature_cols = bundle.get("feature_cols", [])
    target = str(bundle.get("target", TARGET_DEFAULT))
    if not isinstance(feature_cols, list) or not feature_cols:
        raise ValueError(f"Missing feature_cols in {path}")
    return SklearnBundle(model=bundle["model"], feature_cols=[str(c) for c in feature_cols], target=target)


@dataclass(frozen=True)
class XgbBundle:
    """An XGBoost model + feature columns."""

    model: object
    feature_cols: List[str]


def load_xgb_bundle(model_path: Path, meta_path: Path) -> XgbBundle:
    """Load per-station XGBRegressor + its feature column list."""
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
    return XgbBundle(model=model, feature_cols=[str(c) for c in feature_cols])


def to_x(df: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    """Create an X matrix with column reindexing and float32 dtype."""
    x_df = df.reindex(columns=list(feature_cols))
    # Keep NaNs; sklearn pipelines have imputers; XGB handles missing values.
    return x_df.to_numpy(dtype=np.float32, copy=False)


def safe_predict_sklearn(model: object, x: np.ndarray) -> np.ndarray:
    """Predict with a sklearn model, falling back to single-thread if restricted.

    Some restricted Windows environments deny joblib/multiprocessing primitives that
    RandomForest tries to allocate even for prediction. If we hit WinError 5, retry
    with `n_jobs=1` where possible.
    """
    try:
        return np.asarray(model.predict(x), dtype=np.float32)
    except PermissionError:
        # Best-effort: set RF n_jobs=1 and retry.
        try:
            if hasattr(model, "named_steps") and isinstance(getattr(model, "named_steps"), dict):
                steps = getattr(model, "named_steps")
                if "rf" in steps and hasattr(steps["rf"], "n_jobs"):
                    steps["rf"].n_jobs = 1
            elif hasattr(model, "n_jobs"):
                setattr(model, "n_jobs", 1)
        except Exception:
            pass
        return np.asarray(model.predict(x), dtype=np.float32)


def predict_classical_per_station(
    test_df: pd.DataFrame,
    artifacts: Path,
    target: str,
) -> pd.DataFrame:
    """Predict Ridge/RF/XGB per station and return a tidy prediction frame.

    Output rows align 1:1 with test_df rows (where target is not NaN), including:
      station, timestamp, y_true, pred_ridge, pred_rf, pred_xgb
    """
    out_parts: List[pd.DataFrame] = []
    for station, sdf in test_df.groupby("station", sort=False):
        sdf = sdf.dropna(subset=[target]).copy()
        if sdf.empty:
            continue

        paths = station_model_paths(artifacts, str(station))
        ridge = load_joblib_bundle(paths["ridge"])
        rf = load_joblib_bundle(paths["rf"])
        xgb = load_xgb_bundle(paths["xgb"], paths["xgb_meta"])

        x_ridge = to_x(sdf, ridge.feature_cols)
        x_rf = to_x(sdf, rf.feature_cols)
        x_xgb = to_x(sdf, xgb.feature_cols)

        pred_ridge = safe_predict_sklearn(ridge.model, x_ridge)
        pred_rf = safe_predict_sklearn(rf.model, x_rf)
        pred_xgb = np.asarray(xgb.model.predict(x_xgb), dtype=np.float32)

        part = pd.DataFrame(
            {
                "station": sdf["station"].astype(str).to_numpy(copy=False),
                "timestamp": pd.to_datetime(sdf["timestamp"], errors="coerce").to_numpy(copy=False),
                "y_true": pd.to_numeric(sdf[target], errors="coerce").to_numpy(dtype=np.float32, copy=False),
                "pred_ridge": pred_ridge,
                "pred_rf": pred_rf,
                "pred_xgb": pred_xgb,
            }
        )
        out_parts.append(part)

    if not out_parts:
        return pd.DataFrame(columns=["station", "timestamp", "y_true", "pred_ridge", "pred_rf", "pred_xgb"])
    out = pd.concat(out_parts, ignore_index=True, sort=False)
    return out.sort_values(["station", "timestamp"]).reset_index(drop=True)


@dataclass(frozen=True)
class LstmArtifacts:
    """Loaded LSTM model and preprocessing assets."""

    model: object
    feature_cols: List[str]
    x_imputer: object
    x_scaler: object
    y_scaler: object
    seq_len: int


def load_lstm_artifacts(model_path: Path, scalers_path: Path) -> LstmArtifacts:
    """Load LSTM model + scalers produced by `4_train_lstm.py`."""
    import joblib

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not scalers_path.exists():
        raise FileNotFoundError(scalers_path)

    try:
        import tensorflow as tf  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Missing tensorflow for LSTM inference. err={e}")

    model = tf.keras.models.load_model(model_path)
    bundle = joblib.load(scalers_path)
    feature_cols = [str(c) for c in bundle["feature_cols"]]
    return LstmArtifacts(
        model=model,
        feature_cols=feature_cols,
        x_imputer=bundle["x_imputer"],
        x_scaler=bundle["x_scaler"],
        y_scaler=bundle["y_scaler"],
        seq_len=int(bundle.get("seq_len", 96)),
    )


def inverse_transform_target(y_scaled: np.ndarray, y_scaler: object) -> np.ndarray:
    """Inverse transform target scaler into original units."""
    y_scaled_2d = np.asarray(y_scaled, dtype=np.float32).reshape(-1, 1)
    y_inv = y_scaler.inverse_transform(y_scaled_2d).reshape(-1)
    return np.asarray(y_inv, dtype=np.float32)


def lstm_predict_test(
    test_df: pd.DataFrame,
    target: str,
    lstm_model_path: Path,
    lstm_scalers_path: Path,
    batch_size: int,
) -> pd.DataFrame:
    """Predict LSTM on test split, station-safe, returning aligned rows.

    Alignment rules:
      For each station:
        X window uses rows [t-seq_len .. t-1]
        prediction corresponds to timestamp at row t

      Therefore, the first `seq_len` timestamps per station have no prediction.
    """
    try:
        import tensorflow as tf  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Missing tensorflow for LSTM inference. err={e}")

    art = load_lstm_artifacts(lstm_model_path, lstm_scalers_path)
    seq_len = int(art.seq_len)

    parts: List[pd.DataFrame] = []
    for station, sdf in test_df.groupby("station", sort=False):
        sdf = sdf.sort_values("timestamp").dropna(subset=[target]).reset_index(drop=True)
        if sdf.shape[0] <= seq_len + 1:
            continue

        x = sdf.reindex(columns=art.feature_cols).to_numpy(dtype=np.float32, copy=False)
        x_imp = art.x_imputer.transform(x)
        x_scaled = art.x_scaler.transform(x_imp).astype(np.float32, copy=False)

        y = pd.to_numeric(sdf[target], errors="coerce").to_numpy(dtype=np.float32, copy=False).reshape(-1, 1)
        y_scaled = art.y_scaler.transform(y).astype(np.float32, copy=False).reshape(-1)

        # Build station dataset.
        data = x_scaled[:-1]  # (n-1, n_features)
        targets = y_scaled[seq_len:]  # (n-seq_len,)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=targets,
            sequence_length=seq_len,
            sequence_stride=1,
            sampling_rate=1,
            batch_size=int(batch_size),
            shuffle=False,
        )
        # ds yields:
        #   X_batch: (batch, timesteps=seq_len, features=n_features)
        #   y_batch: (batch,) -> we don't need it here
        pred_scaled = np.asarray(art.model.predict(ds, verbose=0), dtype=np.float32).reshape(-1)

        # True targets aligned with predictions:
        y_true_scaled = targets

        pred = inverse_transform_target(pred_scaled, art.y_scaler)
        y_true = inverse_transform_target(y_true_scaled, art.y_scaler)

        # The prediction at index i corresponds to timestamp at row (seq_len + i).
        ts = pd.to_datetime(sdf["timestamp"], errors="coerce").to_numpy(copy=False)
        ts_aligned = ts[seq_len : seq_len + pred.shape[0]]

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
    out = pd.concat(parts, ignore_index=True, sort=False)
    return out.sort_values(["station", "timestamp"]).reset_index(drop=True)


def compute_metrics_table(df: pd.DataFrame, pred_cols: Sequence[str]) -> pd.DataFrame:
    """Compute global + per-station metrics for each prediction column."""
    rows: List[Dict[str, object]] = []

    def _add(scope_station: str, sdf: pd.DataFrame) -> None:
        y = sdf["y_true"].to_numpy(dtype=np.float32, copy=False)
        for col in pred_cols:
            p = sdf[col].to_numpy(dtype=np.float32, copy=False)
            m = evaluate(y, p)
            rows.append(
                {
                    "station": scope_station,
                    "model": col.replace("pred_", ""),
                    "rmse": m["rmse"],
                    "mae": m["mae"],
                    "r2": m["r2"],
                    "n_rows": int(sdf.shape[0]),
                }
            )

    # Global
    _add("ALL", df)
    # Per station
    for station, sdf in df.groupby("station", sort=False):
        _add(str(station), sdf)

    return pd.DataFrame(rows)


def pick_best_tree_model(metrics_global: pd.DataFrame) -> str:
    """Choose best tree model name ('xgb' or 'rf') by global RMSE."""
    trees = metrics_global[metrics_global["model"].isin(["xgb", "rf"])].copy()
    if trees.empty:
        return "xgb"
    trees = trees.sort_values("rmse", ascending=True)
    return str(trees.iloc[0]["model"])


def plot_timeseries_window(
    df_station: pd.DataFrame,
    station: str,
    out_path: Path,
    days: int,
    dpi: int,
) -> None:
    """Plot actual vs predicted for XGB and LSTM for a 3-day window."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sdf = df_station.sort_values("timestamp").reset_index(drop=True)
    if sdf.empty:
        return

    # Choose an early contiguous window of `days` days.
    start = pd.to_datetime(sdf["timestamp"].iloc[0])
    end = start + pd.Timedelta(days=int(days))
    window = sdf[(pd.to_datetime(sdf["timestamp"]) >= start) & (pd.to_datetime(sdf["timestamp"]) < end)].copy()
    if window.empty:
        window = sdf.head(int(days * 24 * 4))  # fallback ~days of 15-min points

    long = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(window["timestamp"], errors="coerce"),
            "Actual": window["y_true"].astype(float),
            "XGBoost": window["pred_xgb"].astype(float),
            "LSTM": window["pred_lstm"].astype(float),
        }
    ).melt(id_vars=["timestamp"], var_name="Series", value_name="PM2.5")

    plt.figure(figsize=(15, 6))
    sns.lineplot(data=long, x="timestamp", y="PM2.5", hue="Series", linewidth=1.6)
    plt.title(f"{station} | PM2.5 | Actual vs Predicted (XGBoost vs LSTM) — {days}-Day Window")
    plt.xlabel("Timestamp")
    plt.ylabel("PM2.5")
    plt.grid(True, which="major", alpha=0.4)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=int(dpi))
    plt.close()


def plot_scatter_comparison(
    df: pd.DataFrame,
    tree_model: str,
    out_path: Path,
    dpi: int,
    max_points: int,
    seed: int,
) -> None:
    """Scatter actual vs predicted for best tree model vs LSTM with y=x reference."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    col_tree = f"pred_{tree_model}"
    if col_tree not in df.columns:
        return

    # Sample points for readability.
    sdf = df[["y_true", col_tree, "pred_lstm"]].dropna().copy()
    if sdf.shape[0] > int(max_points):
        sdf = sdf.sample(n=int(max_points), random_state=int(seed))

    # Determine plot limits.
    lim = float(np.nanmax(sdf[["y_true", col_tree, "pred_lstm"]].to_numpy()))
    lim = max(lim, 1.0)

    plt.figure(figsize=(7.2, 7.2))
    sns.scatterplot(x=sdf["y_true"], y=sdf[col_tree], alpha=0.25, s=18, label=tree_model.upper())
    sns.scatterplot(x=sdf["y_true"], y=sdf["pred_lstm"], alpha=0.25, s=18, label="LSTM")
    plt.plot([0, lim], [0, lim], linestyle="--", color="#2B2B2B", linewidth=1.2, label="y = x")
    plt.title(f"Actual vs Predicted — {tree_model.upper()} vs LSTM")
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.grid(True, which="major", alpha=0.35)
    plt.legend(frameon=True)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=int(dpi))
    plt.close()


def plot_model_comparison_bars(
    metrics_global: pd.DataFrame,
    out_path: Path,
    dpi: int,
) -> None:
    """Bar charts comparing RMSE and R² across all models."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    order = ["ridge", "rf", "xgb", "lstm"]
    mg = metrics_global.copy()
    mg["model"] = mg["model"].astype(str)
    mg = mg[mg["model"].isin(order)].copy()
    mg["model"] = pd.Categorical(mg["model"], categories=order, ordered=True)
    mg = mg.sort_values("model")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.barplot(ax=axes[0], data=mg, x="model", y="rmse")
    axes[0].set_title("RMSE by Model (Test)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(True, axis="y", alpha=0.35)

    sns.barplot(ax=axes[1], data=mg, x="model", y="r2")
    axes[1].set_title("R² by Model (Test)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("R²")
    axes[1].set_ylim(-0.2, 1.0)
    axes[1].grid(True, axis="y", alpha=0.35)

    fig.suptitle("Model Performance Comparison (Global Test Metrics)", y=1.02, fontsize=16, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=int(dpi))
    plt.close()


def main() -> int:
    """Evaluate all models on test set and generate report-ready plots."""
    parser = argparse.ArgumentParser(description="Final evaluation + cinematic plots (Phase E/F).")
    parser.add_argument("--test", type=str, default=str(Path("artifacts") / "features_test_v4.parquet"))
    parser.add_argument("--target", type=str, default=TARGET_DEFAULT)
    parser.add_argument("--artifacts-dir", type=str, default=str(Path("artifacts")))
    parser.add_argument("--lstm-model", type=str, default=str(Path("artifacts") / "model_lstm.keras"))
    parser.add_argument("--lstm-scalers", type=str, default=str(Path("artifacts") / "lstm_scalers.pkl"))
    parser.add_argument("--lstm-batch-size", type=int, default=256)
    parser.add_argument("--timeseries-station", type=str, default="", help="Station for the 3-day time-series plot (default: auto).")
    parser.add_argument("--timeseries-days", type=int, default=3)
    parser.add_argument("--out-metrics", type=str, default=str(Path("artifacts") / "metrics.csv"))
    parser.add_argument("--out-plots", type=str, default=str(Path("artifacts") / "plots"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--max-scatter-points", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    ensure_plot_style()

    artifacts = Path(args.artifacts_dir).resolve()
    out_plots = Path(args.out_plots).resolve()
    out_plots.mkdir(parents=True, exist_ok=True)

    test_df = load_parquet(Path(args.test).resolve()).sort_values(["station", "timestamp"]).reset_index(drop=True)
    target = str(args.target)

    # 1) Predictions for classical per-station models (Ridge/RF/XGB).
    logging.info("Predicting classical models (Ridge/RF/XGB) per station...")
    pred_classical = predict_classical_per_station(test_df=test_df, artifacts=artifacts, target=target)
    if pred_classical.empty:
        logging.error("No classical predictions produced. Did you run `3_train_classical.py --per-station`?")
        return 2

    # 2) Predictions for LSTM (inverse-transformed to original units).
    logging.info("Predicting LSTM on test (station-safe sequences)...")
    pred_lstm = lstm_predict_test(
        test_df=test_df,
        target=target,
        lstm_model_path=Path(args.lstm_model).resolve(),
        lstm_scalers_path=Path(args.lstm_scalers).resolve(),
        batch_size=int(args.lstm_batch_size),
    )
    if pred_lstm.empty:
        logging.error("No LSTM predictions produced. Ensure `artifacts/model_lstm.keras` exists.")
        return 3

    # Align all models on the LSTM-valid timestamps (inner join).
    eval_df = pred_classical.merge(pred_lstm[["station", "timestamp", "pred_lstm"]], on=["station", "timestamp"], how="inner")
    eval_df = eval_df.dropna(subset=["y_true", "pred_ridge", "pred_rf", "pred_xgb", "pred_lstm"]).reset_index(drop=True)
    if eval_df.empty:
        logging.error("No aligned evaluation rows after joining LSTM predictions. Check timestamps and sequence length.")
        return 4

    # 3) Metrics (global + per station)
    pred_cols = ["pred_ridge", "pred_rf", "pred_xgb", "pred_lstm"]
    metrics_df = compute_metrics_table(eval_df, pred_cols=pred_cols)
    out_metrics = Path(args.out_metrics).resolve()
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_metrics, index=False)
    logging.info("Saved metrics: %s", out_metrics)

    # Convenience: global metrics only for plot selection
    metrics_global = metrics_df[metrics_df["station"] == "ALL"].copy()
    best_tree = pick_best_tree_model(metrics_global)
    logging.info("Best tree model (global RMSE) = %s", best_tree)

    # 4) Plots
    # Visual 1: Time-series (3-day window) for a specific station
    station_requested = str(args.timeseries_station).strip()
    if station_requested:
        station_for_ts = station_requested
    else:
        # Prefer a station where XGB is strong in the earlier routing (often AIIMS).
        station_for_ts = str(eval_df["station"].iloc[0])
        if "AIIMS" in set(eval_df["station"].unique()):
            station_for_ts = "AIIMS"
    sdf_ts = eval_df[eval_df["station"] == station_for_ts].copy()
    plot_timeseries_window(
        df_station=sdf_ts,
        station=station_for_ts,
        out_path=out_plots / f"{station_for_ts}_timeseries_3day_xgb_vs_lstm.png",
        days=int(args.timeseries_days),
        dpi=int(args.dpi),
    )

    # Visual 2: Scatter comparison (best tree vs LSTM)
    plot_scatter_comparison(
        df=eval_df,
        tree_model=best_tree,
        out_path=out_plots / f"scatter_{best_tree}_vs_lstm.png",
        dpi=int(args.dpi),
        max_points=int(args.max_scatter_points),
        seed=int(args.seed),
    )

    # Visual 3: Bar chart comparison across all 4 models
    plot_model_comparison_bars(
        metrics_global=metrics_global,
        out_path=out_plots / "model_comparison_bars.png",
        dpi=int(args.dpi),
    )

    logging.info("Plots saved to: %s", out_plots)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
