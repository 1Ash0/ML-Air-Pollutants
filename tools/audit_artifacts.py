from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def _require_imports() -> Tuple[object, object]:
    try:
        import pandas as pd  # type: ignore
        import numpy as np  # type: ignore

        return pd, np
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing deps to audit artifacts. Run this inside your clean venv that has "
            "pandas/numpy/pyarrow installed.\n"
            f"Import error: {e}"
        )


def _file_info(path: Path) -> Dict[str, object]:
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "exists": path.exists(),
        "size_bytes": int(st.st_size),
        "mtime_iso": __import__("datetime").datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
    }


def _check_required_columns(df_columns: List[str], required: List[str]) -> List[str]:
    return [c for c in required if c not in df_columns]


def audit_features(path: Path, target: str = "PM2.5") -> Dict[str, object]:
    pd, _np = _require_imports()

    df = pd.read_parquet(path)
    cols = list(df.columns)

    required = [
        "station",
        "timestamp",
        target,
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        f"{target}_lag_1",
        f"{target}_lag_96",
        f"{target}_rollmean_96",
    ]
    missing_required = _check_required_columns(cols, required)

    pm25_lags = [c for c in cols if c.startswith(f"{target}_lag_")]
    lag_nulls = int(df[[target] + pm25_lags].isna().sum().sum()) if pm25_lags else None
    dup_station_ts = int(df.duplicated(subset=["station", "timestamp"]).sum()) if {"station", "timestamp"} <= set(cols) else None

    station_counts = df.groupby("station").size().sort_values(ascending=False).to_dict() if "station" in cols else {}
    ranges = {}
    if {"station", "timestamp"} <= set(cols):
        rng = df.groupby("station")["timestamp"].agg(["min", "max"])
        ranges = {s: (str(rng.loc[s, "min"]), str(rng.loc[s, "max"])) for s in rng.index}

    return {
        "file": _file_info(path),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing_required_cols": missing_required,
        "n_pm25_lag_cols": int(len(pm25_lags)),
        "nulls_in_target_and_pm25_lags": lag_nulls,
        "dup_station_timestamp": dup_station_ts,
        "rows_by_station": station_counts,
        "range_by_station": ranges,
    }


def audit_scaler(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mean = payload.get("mean", {})
    std = payload.get("std", {})
    return {
        "file": _file_info(path),
        "keys": list(payload.keys()),
        "n_mean": int(len(mean)) if isinstance(mean, dict) else None,
        "n_std": int(len(std)) if isinstance(std, dict) else None,
    }


def main() -> int:
    root = Path(".").resolve()
    art = root / "artifacts"

    train = art / "features_train.parquet"
    val = art / "features_val.parquet"
    test = art / "features_test.parquet"
    scaler = art / "standard_scaler.json"

    out = {
        "features_train": audit_features(train),
        "features_val": audit_features(val),
        "features_test": audit_features(test),
        "standard_scaler": audit_scaler(scaler),
    }

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

