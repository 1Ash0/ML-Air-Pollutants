from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def available_default_targets(df: pd.DataFrame) -> List[str]:
    """Choose a sensible default set of pollutant targets present in the dataset."""
    preferred = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3", "NO", "NOX", "BENZ"]
    present = [c for c in preferred if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not present:
        raise ValueError("No default pollutant targets found in input dataframe.")
    return present


def add_horizon_targets(df: pd.DataFrame, targets: Sequence[str], horizon_steps: int) -> pd.DataFrame:
    """Create forecasting labels y_target(t+h) per station."""
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be >= 1.")
    if "station" not in df.columns:
        raise ValueError("Expected 'station' for station-safe shifting.")
    out = df.copy()
    g = out.groupby("station", sort=False)
    for t in targets:
        out[f"y_{t}_t+{horizon_steps}"] = g[t].shift(-horizon_steps)
    return out


def drop_rows_with_missing_targets(df: pd.DataFrame, y_cols: Sequence[str]) -> pd.DataFrame:
    """Drop rows with missing y labels."""
    return df.dropna(subset=list(y_cols)).copy()


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


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Standard regression metrics."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return {"rmse": metric_rmse(y_true, y_pred), "mae": metric_mae(y_true, y_pred), "r2": metric_r2(y_true, y_pred)}


def get_feature_columns(df: pd.DataFrame, exclude_cols: Sequence[str]) -> List[str]:
    """Select numeric feature columns, excluding identifiers and target columns."""
    exclude = set(META_COLS) | set(exclude_cols)
    cols: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and not df[c].isna().all():
            cols.append(str(c))
    return cols


def build_xy(df: pd.DataFrame, feature_cols: Sequence[str], y_cols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build X, Y, station arrays (aligned)."""
    x = df.reindex(columns=list(feature_cols)).to_numpy(dtype=np.float32, copy=False)
    y = df.reindex(columns=list(y_cols)).to_numpy(dtype=np.float32, copy=False)
    st = df["station"].astype(str).to_numpy()
    return x, y, st


def subsample_by_station(df: pd.DataFrame, max_rows: int, seed: int = 42) -> pd.DataFrame:
    """Station-preserving subsample for speed."""
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    parts: List[pd.DataFrame] = []
    rng = np.random.default_rng(seed)
    stations = df["station"].astype(str)
    for st in sorted(stations.unique()):
        sdf = df.loc[stations == st]
        frac = max_rows / max(len(df), 1)
        n = int(max(1, round(frac * len(sdf))))
        n = min(n, len(sdf))
        idx = rng.choice(sdf.index.to_numpy(), size=n, replace=False)
        parts.append(sdf.loc[idx])
    out = pd.concat(parts, axis=0, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


@dataclass(frozen=True)
class Candidate:
    """A candidate model with predict() producing Y with shape (n, n_targets)."""

    name: str
    kind: str  # "sklearn" | "lstm"
    model: Any


def try_load_saved_model(artifacts: Path, name: str) -> Optional[Any]:
    """Load a previously saved multi-output sklearn model bundle (optional speed-up)."""
    import joblib

    path = artifacts / f"model_{name}.pkl"
    if not path.exists():
        return None
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"Unexpected model bundle format: {path}")
    return bundle["model"]


def train_multi_ridge(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: Sequence[str], y_cols: Sequence[str]) -> Any:
    """Train multi-output Ridge with alpha tuning on validation (mean RMSE)."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x_train, y_train, _ = build_xy(train_df, feature_cols, y_cols)
    x_val, y_val, _ = build_xy(val_df, feature_cols, y_cols)

    best_model: Optional[Any] = None
    best_alpha: Optional[float] = None
    best_score = float("inf")
    for a in [0.1, 1.0, 10.0]:
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("ridge", Ridge(alpha=float(a), fit_intercept=True, random_state=42)),
            ]
        )
        model.fit(x_train, y_train)
        pred = np.asarray(model.predict(x_val), dtype=np.float32)
        rmses = []
        for j in range(len(y_cols)):
            rmses.append(metric_rmse(y_val[:, j], pred[:, j]))
        score = float(np.mean(rmses))
        logging.info("multi_ridge alpha=%s | val_mean_rmse=%.4f", a, score)
        if score < best_score:
            best_score = score
            best_alpha = float(a)
            best_model = model
    if best_model is None or best_alpha is None:
        raise RuntimeError("Failed to train multi_ridge.")
    logging.info("multi_ridge best_alpha=%s", best_alpha)
    return best_model


def train_multi_xgb(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    y_cols: Sequence[str],
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
) -> Any:
    """Train multi-output XGBoost via MultiOutputRegressor."""
    from sklearn.impute import SimpleImputer
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.pipeline import Pipeline
    from xgboost import XGBRegressor

    x_train, y_train, _ = build_xy(train_df, feature_cols, y_cols)
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


def load_multi_lstm(artifacts: Path) -> Tuple[Any, Any, List[str], List[str]]:
    """Load multi-output LSTM model + scalers.

    Returns:
        (keras_model, scalers_dict, feature_cols, y_cols)
    """
    import joblib
    import tensorflow as tf

    model_path = artifacts / "model_multi_lstm.keras"
    scalers_path = artifacts / "multi_lstm_scalers.pkl"
    if not model_path.exists() or not scalers_path.exists():
        raise FileNotFoundError("Missing multi-output LSTM artifacts (model_multi_lstm.keras, multi_lstm_scalers.pkl).")
    model = tf.keras.models.load_model(model_path)
    scalers = joblib.load(scalers_path)
    feature_cols = [str(c) for c in scalers["feature_cols"]]
    y_cols = [str(c) for c in scalers.get("target_cols", [])]
    if not y_cols:
        raise ValueError("multi_lstm_scalers.pkl missing target_cols; re-train with updated script.")
    return model, scalers, feature_cols, y_cols


def make_seq_subset_df(df: pd.DataFrame, seq_len: int, horizon_steps: int) -> pd.DataFrame:
    """Return the split restricted to rows that have enough history for an LSTM window.

    The multi-output LSTM uses sliding windows of length `seq_len` and predicts labels
    that correspond to the end of each window, with an optional horizon.

    For consistency with our LSTM dataset construction:
      earliest usable label index per station = seq_len + horizon_steps - 2

    Args:
        df: Split DataFrame (must contain `station` and y_cols already).
        seq_len: Window length.
        horizon_steps: Forecast horizon in 15-min steps.

    Returns:
        Concatenated subset DataFrame in station-major order.
    """
    start = int(seq_len + horizon_steps - 2)
    if start < 0:
        start = 0
    if "station" not in df.columns:
        raise ValueError("Expected 'station' in df for station-safe slicing.")

    parts: List[pd.DataFrame] = []
    for st in sorted(df["station"].astype(str).unique()):
        sdf = df.loc[df["station"].astype(str) == st]
        if len(sdf) <= start:
            continue
        parts.append(sdf.iloc[start:].copy())
    if not parts:
        raise ValueError("No station had enough rows for the requested seq_len/horizon_steps.")
    return pd.concat(parts, axis=0, ignore_index=True)


def lstm_predict_split_aligned(
    df: pd.DataFrame,
    model: Any,
    scalers: Any,
    feature_cols: Sequence[str],
    y_cols: Sequence[str],
    seq_len: int,
    horizon_steps: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict Y for a split using the multi-output LSTM on the seq-aligned subset.

    Returns arrays aligned to `make_seq_subset_df(df, seq_len, horizon_steps)` ordering:
      - y_true: (n_seq_rows, n_targets)
      - y_pred: (n_seq_rows, n_targets)
      - station: (n_seq_rows,)
    """
    import tensorflow as tf

    # Transform features and targets with saved scalers.
    x = df.reindex(columns=list(feature_cols)).to_numpy(dtype=np.float32, copy=False)
    x_imp = scalers["x_imputer"].transform(x)
    x_scaled = scalers["x_scaler"].transform(x_imp).astype(np.float32, copy=False)

    y = df.reindex(columns=list(y_cols)).to_numpy(dtype=np.float32, copy=False)
    y_scaled = scalers["y_scaler"].transform(y).astype(np.float32, copy=False)

    # Remove non-finite rows to avoid NaNs inside windows.
    mask = np.isfinite(x_scaled).all(axis=1) & np.isfinite(y_scaled).all(axis=1)
    df = df.loc[mask].copy()
    x_scaled = x_scaled[mask]
    y_scaled = y_scaled[mask]

    # Build station-safe dataset and flatten predictions.
    stations = df["station"].astype(str).to_numpy()
    y_true_parts: List[np.ndarray] = []
    y_pred_parts: List[np.ndarray] = []
    st_parts: List[np.ndarray] = []

    for stn in sorted(np.unique(stations)):
        m = stations == stn
        xs = x_scaled[m]
        ys = y_scaled[m]
        n = int(xs.shape[0])
        start = int(seq_len + horizon_steps - 2)
        if n <= start:
            continue

        # Match the training scheme: sequences come from `data` and labels from `targets`.
        data = xs[: n - horizon_steps]
        targets = ys[(horizon_steps - 1) :]

        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=targets,
            sequence_length=int(seq_len),
            sequence_stride=1,
            shuffle=False,
            batch_size=int(batch_size),
        ).prefetch(tf.data.AUTOTUNE)

        y_true_batches: List[np.ndarray] = []
        y_pred_batches: List[np.ndarray] = []
        for x_batch, y_batch in ds:
            pred = model(x_batch, training=False)
            y_true_batches.append(tf.convert_to_tensor(y_batch).numpy())
            y_pred_batches.append(tf.convert_to_tensor(pred).numpy())
        if not y_true_batches:
            continue

        y_true_scaled = np.concatenate(y_true_batches, axis=0).astype(np.float32, copy=False)
        y_pred_scaled = np.concatenate(y_pred_batches, axis=0).astype(np.float32, copy=False)

        # Inverse-transform to original units.
        y_true = scalers["y_scaler"].inverse_transform(y_true_scaled).astype(np.float32, copy=False)
        y_pred = scalers["y_scaler"].inverse_transform(y_pred_scaled).astype(np.float32, copy=False)

        y_true_parts.append(y_true)
        y_pred_parts.append(y_pred)
        st_parts.append(np.asarray([stn] * y_true.shape[0], dtype=object))

    if not y_true_parts:
        raise ValueError("No station produced LSTM sequences for this split.")
    y_true_all = np.concatenate(y_true_parts, axis=0)
    y_pred_all = np.concatenate(y_pred_parts, axis=0)
    st_all = np.concatenate(st_parts, axis=0).astype(str)
    return y_true_all, y_pred_all, st_all


def evaluate_per_station_target(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_cols: Sequence[str],
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, float]]]:
    """Compute metrics per station and per target, plus overall per target."""
    stations = df["station"].astype(str).to_numpy()
    per_station: Dict[str, Dict[str, Dict[str, float]]] = {}
    overall: Dict[str, Dict[str, float]] = {}

    # Overall per target
    for j, t in enumerate(y_cols):
        overall[str(t)] = evaluate(y_true[:, j], y_pred[:, j])

    # Per station per target
    for st in np.unique(stations):
        m = stations == st
        if int(np.sum(m)) < 50:
            continue
        per_station[str(st)] = {}
        for j, t in enumerate(y_cols):
            per_station[str(st)][str(t)] = evaluate(y_true[m, j], y_pred[m, j])
    return per_station, overall


def route_by_station_target(
    stations: Sequence[str],
    y_cols: Sequence[str],
    val_scores: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
) -> Dict[str, Dict[str, str]]:
    """Choose best model per (station, target) by lowest validation RMSE."""
    routing: Dict[str, Dict[str, str]] = {}
    for st in stations:
        routing[str(st)] = {}
        for t in y_cols:
            best_name: Optional[str] = None
            best_rmse: Optional[float] = None
            for model_name, scores in val_scores.items():
                rmse = scores.get(str(st), {}).get(str(t), {}).get("rmse")
                if rmse is None or (isinstance(rmse, float) and np.isnan(rmse)):
                    continue
                r = float(rmse)
                if best_rmse is None or r < best_rmse:
                    best_rmse = r
                    best_name = str(model_name)
            if best_name is None:
                best_name = "multi_ridge"
            routing[str(st)][str(t)] = best_name
    return routing


def write_routed_outputs(
    artifacts: Path,
    y_cols: Sequence[str],
    routing: Dict[str, Dict[str, str]],
    val_scores: Dict[str, Any],
    test_scores: Dict[str, Any],
    out_json: Path,
    out_csv: Path,
) -> None:
    """Write routed JSON + routed metrics CSV."""
    # Build CSV rows (ALL + per station per target).
    rows: List[Dict[str, Any]] = []
    # Per station per target metrics
    for st, per_t in test_scores["per_station"].items():
        for t, m in per_t.items():
            rows.append({"station": st, "model": "ROUTED", "target": t, **m, "n_rows": 0})
    # Overall per target
    for t, m in test_scores["overall"].items():
        rows.append({"station": "ALL", "model": "ROUTED", "target": t, **m, "n_rows": int(test_scores.get("n_rows", 0))})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logging.info("Wrote: %s", out_csv)

    payload = {
        "horizon_minutes": int(15),
        "routing_rule": "lowest validation RMSE per (station, target)",
        "targets": list(y_cols),
        "routing_table": routing,
        "val_scores": val_scores,
        "test_scores": test_scores,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Wrote: %s", out_json)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Route best multi-output model per station×pollutant target using validation RMSE."
    )
    parser.add_argument("--train", type=str, default=str(Path("artifacts") / "features_train_v4.parquet"))
    parser.add_argument("--val", type=str, default=str(Path("artifacts") / "features_val_v4.parquet"))
    parser.add_argument("--test", type=str, default=str(Path("artifacts") / "features_test_v4.parquet"))
    parser.add_argument("--artifacts", type=str, default=str(Path("artifacts")))
    parser.add_argument("--targets", type=str, default="", help="Comma-separated raw pollutant targets; default autodetect.")
    parser.add_argument("--horizon-steps", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--xgb-max-train-rows", type=int, default=50_000)
    parser.add_argument("--xgb-n-estimators", type=int, default=400)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--skip-xgb", action="store_true")
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--out-json", type=str, default=str(Path("artifacts") / "multioutput_routed.json"))
    parser.add_argument("--out-csv", type=str, default=str(Path("artifacts") / "multioutput_metrics_routed.csv"))
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    artifacts = Path(args.artifacts)

    train_path = resolve_input_path(Path(args.train), "features_train_v4.parquet")
    val_path = resolve_input_path(Path(args.val), "features_val_v4.parquet")
    test_path = resolve_input_path(Path(args.test), "features_test_v4.parquet")

    df_train = load_parquet(train_path)
    df_val = load_parquet(val_path)
    df_test = load_parquet(test_path)

    raw_targets = [t.strip() for t in args.targets.split(",") if t.strip()] if args.targets else available_default_targets(df_train)
    horizon = int(args.horizon_steps)
    y_cols = [f"y_{t}_t+{horizon}" for t in raw_targets]

    # Build explicit forecast labels and remove rows without labels.
    df_train = drop_rows_with_missing_targets(add_horizon_targets(df_train, targets=raw_targets, horizon_steps=horizon), y_cols=y_cols)
    df_val = drop_rows_with_missing_targets(add_horizon_targets(df_val, targets=raw_targets, horizon_steps=horizon), y_cols=y_cols)
    df_test = drop_rows_with_missing_targets(add_horizon_targets(df_test, targets=raw_targets, horizon_steps=horizon), y_cols=y_cols)

    # Feature columns shared by Ridge/XGB routing candidates.
    feature_cols = get_feature_columns(df_train, exclude_cols=list(raw_targets) + list(y_cols))
    logging.info("Targets=%d | n_features=%d | horizon_steps=%d", len(y_cols), len(feature_cols), horizon)

    # Optional training cap for XGB speed.
    if int(args.xgb_max_train_rows) > 0:
        df_train_xgb = subsample_by_station(df_train, max_rows=int(args.xgb_max_train_rows), seed=42)
        logging.info("Subsampled XGB train => %d rows", len(df_train_xgb))
    else:
        df_train_xgb = df_train

    # Train candidates.
    candidates: List[Candidate] = []
    ridge = try_load_saved_model(artifacts, "multi_ridge") or train_multi_ridge(
        df_train, df_val, feature_cols=feature_cols, y_cols=y_cols
    )
    candidates.append(Candidate(name="multi_ridge", kind="sklearn", model=ridge))

    xgb = None
    if not args.skip_xgb:
        logging.info("Training multi_xgb (for routing candidates)...")
        xgb = try_load_saved_model(artifacts, "multi_xgb") or train_multi_xgb(
            df_train_xgb,
            feature_cols=feature_cols,
            y_cols=y_cols,
            n_estimators=int(args.xgb_n_estimators),
            learning_rate=float(args.xgb_learning_rate),
            max_depth=int(args.xgb_max_depth),
        )
        candidates.append(Candidate(name="multi_xgb", kind="sklearn", model=xgb))

    lstm_bundle = None
    if not args.skip_lstm:
        try:
            model_lstm, scalers, lstm_feature_cols, lstm_y_cols = load_multi_lstm(artifacts)
            # Ensure target cols match routing targets.
            if list(lstm_y_cols) != list(y_cols):
                raise ValueError("multi_lstm target_cols mismatch; re-train multi LSTM with same targets/horizon.")
            lstm_bundle = (model_lstm, scalers, lstm_feature_cols, lstm_y_cols)
            candidates.append(Candidate(name="multi_lstm", kind="lstm", model=model_lstm))
        except Exception as e:
            logging.warning("Skipping multi_lstm candidate (not available): %s", e)

    stations = sorted(df_val["station"].astype(str).unique())

    # Use a seq-aligned subset for fair routing when including LSTM:
    # all candidates are evaluated on the same (station, time) rows that have enough history.
    df_val_seq = make_seq_subset_df(df_val, seq_len=int(args.seq_len), horizon_steps=horizon)
    df_test_seq = make_seq_subset_df(df_test, seq_len=int(args.seq_len), horizon_steps=horizon)

    # Evaluate candidates on VAL (per station×target metrics) using seq-aligned subset.
    val_scores: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for cand in candidates:
        if cand.kind == "sklearn":
            x_val, y_val, _st_val = build_xy(df_val_seq, feature_cols, y_cols)
            pred = np.asarray(cand.model.predict(x_val), dtype=np.float32)
            # Use df_val station column for grouping.
            per_station, _overall = evaluate_per_station_target(df_val_seq, y_val, pred, y_cols=y_cols)
            val_scores[cand.name] = per_station
        else:
            if lstm_bundle is None:
                continue
            model_lstm, scalers, fcols, _ = lstm_bundle
            y_true, y_pred, st_seq = lstm_predict_split_aligned(
                df=df_val,
                model=model_lstm,
                scalers=scalers,
                feature_cols=fcols,
                y_cols=y_cols,
                seq_len=int(args.seq_len),
                horizon_steps=horizon,
                batch_size=int(args.batch_size),
            )
            df_l = df_val_seq.iloc[: y_true.shape[0]].copy()
            df_l["station"] = st_seq  # exact station labels aligned to predictions
            per_station, _overall = evaluate_per_station_target(df_l, y_true, y_pred, y_cols=y_cols)
            val_scores[cand.name] = per_station

    routing = route_by_station_target(stations=stations, y_cols=y_cols, val_scores=val_scores)

    # Evaluate on TEST (seq-aligned) and apply routing.
    preds_test: Dict[str, np.ndarray] = {}
    x_test, y_test_mat, st_test = build_xy(df_test_seq, feature_cols, y_cols)
    for cand in candidates:
        if cand.kind == "sklearn":
            preds_test[cand.name] = np.asarray(cand.model.predict(x_test), dtype=np.float32)
        else:
            if lstm_bundle is None:
                continue
            model_lstm, scalers, fcols, _ = lstm_bundle
            y_true_l, y_pred_l, st_l = lstm_predict_split_aligned(
                df=df_test,
                model=model_lstm,
                scalers=scalers,
                feature_cols=fcols,
                y_cols=y_cols,
                seq_len=int(args.seq_len),
                horizon_steps=horizon,
                batch_size=int(args.batch_size),
            )
            # Align to seq subset length (they should match; truncate defensively).
            n = int(min(y_test_mat.shape[0], y_pred_l.shape[0]))
            y_test_mat = y_test_mat[:n]
            st_test = st_test[:n]
            preds_test[cand.name] = y_pred_l[:n]
            preds_test["multi_lstm__y_true"] = y_true_l[:n]

    # Routed prediction matrix
    y_pred_routed = np.zeros_like(y_test_mat, dtype=np.float32)
    for st in np.unique(st_test):
        mask = st_test == st
        for j, t in enumerate(y_cols):
            chosen = routing.get(str(st), {}).get(str(t), "multi_ridge")
            if chosen not in preds_test:
                chosen = "multi_ridge"
            y_pred_routed[mask, j] = preds_test[chosen][mask, j]

    # Metrics per station×target and overall per target.
    per_station, overall = evaluate_per_station_target(
        df=df_test_seq.iloc[: y_test_mat.shape[0]].assign(station=st_test),  # ensure aligned
        y_true=y_test_mat,
        y_pred=y_pred_routed,
        y_cols=y_cols,
    )
    test_scores = {"per_station": per_station, "overall": overall, "n_rows": int(y_test_mat.shape[0])}

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    write_routed_outputs(
        artifacts=artifacts,
        y_cols=y_cols,
        routing=routing,
        val_scores=val_scores,
        test_scores=test_scores,
        out_json=out_json,
        out_csv=out_csv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
