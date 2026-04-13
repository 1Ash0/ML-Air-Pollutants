from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file with a clear error message."""
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet: {path}")
    return pd.read_parquet(path)


def df_summary(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    """Compute a compact dataset summary."""
    summary: Dict[str, Any] = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "stations": sorted([str(s) for s in df["station"].dropna().unique()]) if "station" in df.columns else [],
    }

    if target in df.columns:
        s = pd.to_numeric(df[target], errors="coerce")
        summary["target"] = {
            "present": True,
            "nan": int(s.isna().sum()),
            "min": float(np.nanmin(s.to_numpy())) if s.notna().any() else None,
            "max": float(np.nanmax(s.to_numpy())) if s.notna().any() else None,
            "neg": int((s < 0).sum(skipna=True)),
            "gt_1e5": int((s > 1e5).sum(skipna=True)),
        }
    else:
        summary["target"] = {"present": False}

    if "station" in df.columns:
        summary["rows_by_station"] = {str(k): int(v) for k, v in df["station"].value_counts(dropna=False).items()}
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick sanity checks for feature split parquet files.")
    parser.add_argument("--train", type=str, default=str(Path("artifacts") / "features_train.parquet"))
    parser.add_argument("--val", type=str, default=str(Path("artifacts") / "features_val.parquet"))
    parser.add_argument("--test", type=str, default=str(Path("artifacts") / "features_test.parquet"))
    parser.add_argument("--target", type=str, default="PM2.5")
    parser.add_argument("--out-json", type=str, default=str(Path("artifacts") / "feature_splits_report.json"))
    args = parser.parse_args()

    train_p = Path(args.train).resolve()
    val_p = Path(args.val).resolve()
    test_p = Path(args.test).resolve()

    train = load_parquet(train_p)
    val = load_parquet(val_p)
    test = load_parquet(test_p)

    target = str(args.target)

    train_cols = set(train.columns)
    val_cols = set(val.columns)
    test_cols = set(test.columns)

    report: Dict[str, Any] = {
        "paths": {"train": str(train_p), "val": str(val_p), "test": str(test_p)},
        "target": target,
        "train": df_summary(train, target),
        "val": df_summary(val, target),
        "test": df_summary(test, target),
        "schema_diff": {
            "val_missing_from_train": sorted(list(train_cols - val_cols)),
            "val_extra_vs_train": sorted(list(val_cols - train_cols)),
            "test_missing_from_train": sorted(list(train_cols - test_cols)),
            "test_extra_vs_train": sorted(list(test_cols - train_cols)),
        },
    }

    out_p = Path(args.out_json).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote: {out_p}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise

