from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def setup_logging(level: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_plot_style() -> None:
    """Configure seaborn/matplotlib for a clean academic look."""
    sns.set_theme(style="whitegrid", context="talk")
    sns.set_palette("deep")
    plt.rcParams.update(
        {
            "axes.facecolor": "#FBFBFB",
            "figure.facecolor": "white",
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.8,
            "axes.edgecolor": "#2B2B2B",
            "axes.linewidth": 1.0,
            "axes.titleweight": "bold",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def save_fig(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    """Save a figure to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    logging.info("Saved: %s", path)


def load_metrics(path: Path) -> pd.DataFrame:
    """Load the `multioutput_metrics.csv` file."""
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"station", "model", "target", "rmse", "mae", "r2", "n_rows"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df


def plot_overall_heatmaps(df: pd.DataFrame, out_dir: Path) -> None:
    """Heatmaps of RMSE and R² by (target × model) on station=ALL."""
    sdf = df.loc[df["station"] == "ALL"].copy()
    if sdf.empty:
        logging.warning("No station=ALL rows found; skipping overall heatmaps.")
        return

    # R² heatmap
    piv_r2 = sdf.pivot_table(index="target", columns="model", values="r2", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, max(5, 0.55 * len(piv_r2))))
    sns.heatmap(piv_r2, annot=True, fmt=".3f", cmap="RdYlGn", vmin=-1.0, vmax=1.0, linewidths=0.6, ax=ax)
    ax.set_title("Multi-Output Forecasting (t+15 min): R² by Target and Model (Test)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Pollutant Target")
    save_fig(fig, out_dir / "multioutput_r2_heatmap_all.png")

    # RMSE heatmap
    piv_rmse = sdf.pivot_table(index="target", columns="model", values="rmse", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, max(5, 0.55 * len(piv_rmse))))
    sns.heatmap(piv_rmse, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.6, ax=ax)
    ax.set_title("Multi-Output Forecasting (t+15 min): RMSE by Target and Model (Test)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Pollutant Target")
    save_fig(fig, out_dir / "multioutput_rmse_heatmap_all.png")


def plot_station_target_heatmap(df: pd.DataFrame, out_dir: Path, model: Optional[str] = None) -> None:
    """Heatmap of R² per station vs target for a chosen multi-output model."""
    sdf = df.loc[df["station"] != "ALL"].copy()
    if model is not None:
        sdf = sdf.loc[sdf["model"] == model]
    if sdf.empty:
        logging.warning("No per-station rows found for model=%s; skipping.", model)
        return

    # Pick best model by PM2.5 R² (if present) else mean R² if not specified.
    if model is None:
        pm25_keys = [t for t in sdf["target"].astype(str).unique() if "PM2.5" in t and "t+" in t]
        pm25_key = pm25_keys[0] if pm25_keys else None
        if pm25_key is not None:
            model_scores = (
                sdf.loc[sdf["target"].astype(str) == pm25_key]
                .groupby("model")["r2"]
                .mean()
                .sort_values(ascending=False)
            )
            logging.info("Auto-selected best multi-output model by %s mean R²: %s", pm25_key, model_scores.index[0])
        else:
            model_scores = sdf.groupby("model")["r2"].mean().sort_values(ascending=False)
            logging.info("Auto-selected best multi-output model by mean R²: %s", model_scores.index[0])
        model = str(model_scores.index[0])
        sdf = sdf.loc[sdf["model"] == model]

    piv = sdf.pivot_table(index="station", columns="target", values="r2", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(12, 0.9 * len(piv.columns)), 5))
    sns.heatmap(piv, annot=True, fmt=".3f", cmap="RdYlGn", vmin=-1.0, vmax=1.0, linewidths=0.6, ax=ax)
    ax.set_title(f"Multi-Output Forecasting (t+15 min): Station × Target R² (Model={model})")
    ax.set_xlabel("Pollutant Target")
    ax.set_ylabel("Station")
    save_fig(fig, out_dir / f"multioutput_station_target_r2_{model}.png")


def plot_station_target_heatmaps_all_models(df: pd.DataFrame, out_dir: Path) -> None:
    """Write station×target R² heatmap for every model present."""
    sdf = df.loc[df["station"] != "ALL"].copy()
    if sdf.empty:
        return
    for model in sorted(sdf["model"].astype(str).unique()):
        plot_station_target_heatmap(df, out_dir=out_dir, model=str(model))


def plot_target_bars(df: pd.DataFrame, out_dir: Path) -> None:
    """Bar charts comparing model RMSE/R² per target (station=ALL)."""
    sdf = df.loc[df["station"] == "ALL"].copy()
    if sdf.empty:
        return

    # RMSE bars
    fig, ax = plt.subplots(figsize=(14, max(6, 0.6 * sdf["target"].nunique())))
    sns.barplot(data=sdf, y="target", x="rmse", hue="model", ax=ax)
    ax.set_title("Multi-Output Forecasting (t+15 min): RMSE per Pollutant Target (Test)")
    ax.set_xlabel("RMSE")
    ax.set_ylabel("Pollutant Target")
    ax.grid(True, axis="x", alpha=0.3)
    save_fig(fig, out_dir / "multioutput_rmse_bars_all.png")

    # R² bars
    fig, ax = plt.subplots(figsize=(14, max(6, 0.6 * sdf["target"].nunique())))
    sns.barplot(data=sdf, y="target", x="r2", hue="model", ax=ax)
    ax.set_title("Multi-Output Forecasting (t+15 min): R² per Pollutant Target (Test)")
    ax.set_xlabel("R²")
    ax.set_ylabel("Pollutant Target")
    ax.axvline(0.0, color="#2B2B2B", linewidth=1.0, alpha=0.7)
    ax.grid(True, axis="x", alpha=0.3)
    save_fig(fig, out_dir / "multioutput_r2_bars_all.png")


def plot_routed_heatmaps(routed_csv: Path, out_dir: Path) -> None:
    """Heatmaps for the routed multi-output system (if present)."""
    if not routed_csv.exists():
        return
    df = pd.read_csv(routed_csv)
    sdf = df.loc[df["station"] != "ALL"].copy()
    if sdf.empty:
        return

    piv_r2 = sdf.pivot_table(index="station", columns="target", values="r2", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(12, 0.9 * len(piv_r2.columns)), 5))
    sns.heatmap(piv_r2, annot=True, fmt=".3f", cmap="RdYlGn", vmin=-1.0, vmax=1.0, linewidths=0.6, ax=ax)
    ax.set_title("Routed Multi-Output System: Station × Target R² (Test)\nRouting criterion: lowest validation RMSE per (station,target)")
    ax.set_xlabel("Pollutant Target")
    ax.set_ylabel("Station")
    save_fig(fig, out_dir / "multioutput_routed_station_target_r2.png")

    sdf_all = df.loc[df["station"] == "ALL"].copy()
    if not sdf_all.empty:
        piv_rmse = sdf_all.pivot_table(index="target", values="rmse", aggfunc="mean").sort_values("rmse", ascending=False)
        fig, ax = plt.subplots(figsize=(12, max(6, 0.4 * len(piv_rmse))))
        sns.barplot(data=piv_rmse.reset_index(), x="rmse", y="target", ax=ax, color="#1B5E20")
        ax.set_title("Routed Multi-Output System: RMSE per Target (ALL Stations, Test)")
        ax.set_xlabel("RMSE")
        ax.set_ylabel("Target")
        ax.grid(True, axis="x", alpha=0.3)
        save_fig(fig, out_dir / "multioutput_routed_rmse_per_target_all.png")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate viva-ready plots for multi-output pollutant forecasts.")
    parser.add_argument("--metrics-csv", type=str, default=str(Path("artifacts") / "multioutput_metrics.csv"))
    parser.add_argument("--out-dir", type=str, default=str(Path("artifacts") / "plots" / "viva" / "07_multioutput"))
    parser.add_argument("--routed-csv", type=str, default=str(Path("artifacts") / "multioutput_metrics_routed.csv"))
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    ensure_plot_style()

    metrics_path = Path(args.metrics_csv)
    out_dir = Path(args.out_dir)
    df = load_metrics(metrics_path)

    plot_overall_heatmaps(df, out_dir=out_dir)
    plot_target_bars(df, out_dir=out_dir)
    plot_station_target_heatmap(df, out_dir=out_dir, model=None)
    plot_station_target_heatmaps_all_models(df, out_dir=out_dir)
    plot_routed_heatmaps(Path(args.routed_csv), out_dir=out_dir)

    logging.info("Done. Multi-output plots written to: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
