from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit raw timestamp spacing and sheet sources in Phase A parquet.")
    parser.add_argument("--in-parquet", type=str, default=str(Path("artifacts") / "data_15min.parquet"))
    parser.add_argument("--top-sheets", type=int, default=20)
    parser.add_argument("--sheet-audit", action="store_true", help="Audit delta minutes within each (station, source_sheet).")
    args = parser.parse_args()

    in_path = Path(args.in_parquet).resolve()
    df = pd.read_parquet(in_path, columns=["station", "timestamp", "source_sheet"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["station", "timestamp"]).copy()

    print(f"File: {in_path}")
    print(f"Rows: {len(df):,}")
    print(f"Stations: {sorted(df['station'].astype(str).unique().tolist())}")

    print("\nTop source_sheet overall:")
    print(df["source_sheet"].astype(str).value_counts().head(int(args.top_sheets)).to_string())

    print("\nPer-station raw delta (minutes) distribution (top 8):")
    for station, sdf in df.groupby("station", sort=False):
        sdf = sdf.sort_values("timestamp")
        deltas = sdf["timestamp"].diff().dt.total_seconds().dropna()
        if deltas.empty:
            continue
        mins = (deltas / 60.0).round().astype(int)
        vc = mins.value_counts().head(8)
        frac_15 = float(np.isclose(deltas.to_numpy(), 900.0).mean())
        frac_60 = float(np.isclose(deltas.to_numpy(), 3600.0).mean())
        print(f"\nStation={station}")
        print(vc.to_string())
        print(
            f"frac_15min={frac_15:.3f} frac_60min={frac_60:.3f} "
            f"| span {sdf['timestamp'].min()} -> {sdf['timestamp'].max()} | unique_sheets={sdf['source_sheet'].nunique(dropna=False)}"
        )

    if bool(args.sheet_audit):
        print("\nPer (station, source_sheet) delta audit (showing sheets with notable 60-min steps):")
        out_rows = []
        for (station, sheet), sdf in df.groupby(["station", "source_sheet"], sort=False):
            sdf = sdf.sort_values("timestamp")
            deltas = sdf["timestamp"].diff().dt.total_seconds().dropna()
            if deltas.empty:
                continue
            frac_60 = float(np.isclose(deltas.to_numpy(), 3600.0).mean())
            frac_15 = float(np.isclose(deltas.to_numpy(), 900.0).mean())
            out_rows.append(
                {
                    "station": str(station),
                    "sheet": str(sheet),
                    "rows": int(sdf.shape[0]),
                    "frac_15": frac_15,
                    "frac_60": frac_60,
                }
            )
        odf = pd.DataFrame(out_rows)
        if not odf.empty:
            # Show sheets where hourly steps are common (>=10%) and have enough rows.
            focus = odf[(odf["rows"] >= 500) & (odf["frac_60"] >= 0.10)].sort_values(["frac_60", "rows"], ascending=False)
            if focus.empty:
                print("No sheets found with >=10% hourly deltas (rows>=500).")
            else:
                print(focus.head(50).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
