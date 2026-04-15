"""
6_viva_plots.py  — Comprehensive Viva-Ready Visualisations
===========================================================
Generates 20+ publication-quality plots from:
  - Parquet test/train data      (artifacts/features_test_v4.parquet, features_train_v4.parquet)
  - JSON metric artifacts        (classical_metrics_per_station.json,
                                  classical_metrics_routed_v4.json, lstm_metrics.json)
  - Metrics CSV                  (artifacts/metrics.csv)

All plots are self-explanatory with descriptive titles and annotations.

Output layout:
  artifacts/plots/viva/
    01_data_overview/
    02_inter_station/
    03_correlation/
    04_model_performance/
    05_lstm/
    06_routing/

Run:
    python 6_viva_plots.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ARTIFACTS          = Path("artifacts")
PARQUET_TRAIN      = ARTIFACTS / "features_train_v4.parquet"
PARQUET_TEST       = ARTIFACTS / "features_test_v4.parquet"
METRICS_CSV        = ARTIFACTS / "metrics.csv"
JSON_PER_STATION   = ARTIFACTS / "classical_metrics_per_station.json"
JSON_ROUTED        = ARTIFACTS / "classical_metrics_routed_v4.json"
JSON_LSTM          = ARTIFACTS / "lstm_metrics.json"
OUT_DIR            = ARTIFACTS / "plots" / "viva"

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
STATIONS = ["AIIMS", "BHATAGAON", "IGKV", "SILTARA"]

STATION_COLORS = {
    "AIIMS":     "#1976D2",
    "BHATAGAON": "#388E3C",
    "IGKV":      "#F57C00",
    "SILTARA":   "#C2185B",
}

MODEL_COLORS = {
    "RIDGE":  "#1565C0",
    "RF":     "#B71C1C",
    "XGB":    "#2E7D32",
    "LSTM":   "#6A1B9A",
    "ROUTED": "#E65100",
}

# Core pollutant/meteorological columns (without lag/rolling suffixes)
CORE_COLS = [
    "PM2.5", "PM10", "CO", "NO", "NO2", "NOX",
    "SO2", "O3", "NH3", "BENZ", "TEMP", "HUM", "WS", "SR", "RG",
]

DPI = 200


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _setup() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "#F8F9FA",
        "axes.edgecolor":    "#2B2B2B",
        "axes.linewidth":    1.1,
        "axes.titleweight":  "bold",
        "axes.titlesize":    13,
        "axes.labelsize":    12,
        "font.family":       "DejaVu Sans",
        "grid.color":        "#DDDDDD",
        "grid.linewidth":    0.8,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.12,
        "legend.framealpha": 0.9,
    })


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    logging.info("Saved: %s", path.relative_to(Path(".")))


def _load_parquet(p: Path, ts_parse: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(p)
    if ts_parse and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 01  DATA OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

def plot_dataset_size(out: Path) -> None:
    """
    Grouped bar chart showing train/val/test record counts per station.
    Confirms the 70/15/15 chronological split strategy.
    """
    rows = {
        "Train": {"AIIMS": 78672, "BHATAGAON": 72186, "IGKV": 60566, "SILTARA": 68146},
        "Val":   {"AIIMS": 17515, "BHATAGAON": 17241, "IGKV": 14418, "SILTARA": 15906},
        "Test":  {"AIIMS": 18027, "BHATAGAON": 18289, "IGKV": 14294, "SILTARA": 14452},
    }
    splits  = list(rows.keys())
    colors  = ["#1565C0", "#0288D1", "#4FC3F7"]
    x       = np.arange(len(STATIONS))
    width   = 0.26

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (split, color) in enumerate(zip(splits, colors)):
        vals = [rows[split][s] for s in STATIONS]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=split,
                      color=color, edgecolor="white", linewidth=0.8)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 600,
                    f"{h:,}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(STATIONS, fontsize=13)
    ax.set_xlabel("Monitoring Station")
    ax.set_ylabel("15-min Records")
    ax.set_title(
        "Dataset Size per Station and Chronological Split\n"
        "Strategy: 70 % Train → 15 % Val → 15 % Test (no shuffling — respects time order)"
    )
    ax.legend(title="Split", frameon=True)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_ylim(0, 100_000)
    _save(fig, out)


def plot_pm25_violin(df: pd.DataFrame, out: Path) -> None:
    """
    Violin + box overlay showing full PM2.5 distribution per station.
    Reveals spread, outliers and median differences — key context for viva.
    """
    data = df[["station", "PM2.5"]].dropna()

    fig, ax = plt.subplots(figsize=(13, 7))
    sns.violinplot(data=data, x="station", y="PM2.5", order=STATIONS,
                   palette=STATION_COLORS, inner=None, ax=ax, alpha=0.55)
    sns.boxplot(data=data, x="station", y="PM2.5", order=STATIONS,
                palette=STATION_COLORS, width=0.14, showcaps=True,
                flierprops=dict(marker=".", markersize=2, alpha=0.25),
                ax=ax, linewidth=1.2, boxprops=dict(alpha=0.85))

    for i, stn in enumerate(STATIONS):
        sub  = data[data["station"] == stn]["PM2.5"]
        med  = sub.median()
        mean = sub.mean()
        std  = sub.std()
        ax.text(i,  med + 2,  f"Med={med:.1f}",  ha="center", va="bottom", fontsize=9,  fontweight="bold")
        ax.text(i, -7,        f"μ={mean:.1f}\nσ={std:.1f}", ha="center", va="top",    fontsize=8.5, color="#555")

    ax.set_xlabel("Monitoring Station")
    ax.set_ylabel("PM2.5 Concentration (μg/m³)")
    ax.set_title(
        "PM2.5 Distribution by Station (Training Data)\n"
        "Violin = kernel density; Box = IQR + median; IGKV lowest pollution, SILTARA highest spread"
    )
    ax.set_ylim(-12, None)
    _save(fig, out)


def plot_pm25_by_year(df: pd.DataFrame, out: Path) -> None:
    """
    Box plot of PM2.5 grouped by station and year — reveals inter-annual trends.
    """
    data = df[["station", "timestamp", "PM2.5"]].dropna().copy()
    data["year"] = data["timestamp"].dt.year.astype(str)

    fig, ax = plt.subplots(figsize=(15, 6))
    sns.boxplot(data=data, x="station", y="PM2.5", hue="year", order=STATIONS,
                palette="Set2", width=0.65,
                flierprops=dict(marker=".", markersize=2, alpha=0.25),
                linewidth=0.9, ax=ax)
    ax.set_xlabel("Monitoring Station")
    ax.set_ylabel("PM2.5 (μg/m³)")
    ax.set_title(
        "Year-wise PM2.5 Distribution per Station\n"
        "Reveals inter-annual variability and data availability per year"
    )
    ax.legend(title="Year", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
    _save(fig, out)


# ─────────────────────────────────────────────────────────────────────────────
# 02  INTER-STATION PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

def plot_hourly_heatmap(df: pd.DataFrame, out: Path) -> None:
    """
    Heatmap: mean PM2.5 by hour-of-day × station.
    Highlights daily pollution peaks (morning rush, night inversion).
    """
    data = df[["station", "timestamp", "PM2.5"]].dropna().copy()
    data["hour"] = data["timestamp"].dt.hour
    pivot = (data.groupby(["hour", "station"])["PM2.5"]
                 .mean()
                 .unstack("station")
                 .reindex(columns=STATIONS))

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.4, linecolor="white",
                fmt=".1f", annot=True, annot_kws={"size": 8.5},
                cbar_kws={"label": "Mean PM2.5 (μg/m³)"}, ax=ax)
    ax.set_xlabel("Monitoring Station")
    ax.set_ylabel("Hour of Day  (0 = midnight, 12 = noon)")
    ax.set_title(
        "Diurnal PM2.5 Pattern by Station (Training Data)\n"
        "Identifies peak pollution hours — critical for model feature design"
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    _save(fig, out)


def plot_dow_pattern(df: pd.DataFrame, out: Path) -> None:
    """
    Line chart: mean PM2.5 by day-of-week × station.
    Tests whether weekend traffic reduction affects PM2.5.
    """
    data = df[["station", "timestamp", "PM2.5"]].dropna().copy()
    data["dow"] = data["timestamp"].dt.dayofweek
    agg  = data.groupby(["dow", "station"])["PM2.5"].mean().reset_index()
    DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for stn in STATIONS:
        s = agg[agg["station"] == stn].sort_values("dow")
        ax.plot(s["dow"], s["PM2.5"], marker="o", label=stn,
                color=STATION_COLORS[stn], linewidth=2.2, markersize=7)

    ax.set_xticks(range(7))
    ax.set_xticklabels(DAYS)
    ax.axvspan(4.5, 6.5, alpha=0.08, color="gray")
    ymax = ax.get_ylim()[1]
    ax.text(5.5, ymax * 0.97, "Weekend", ha="center", fontsize=10, color="gray", style="italic")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Mean PM2.5 (μg/m³)")
    ax.set_title(
        "Weekly PM2.5 Pattern by Station\n"
        "Shaded area = weekend — minimal weekend dip suggests non-traffic dominant sources"
    )
    ax.legend(title="Station", frameon=True)
    _save(fig, out)


def plot_monthly_pattern(df: pd.DataFrame, out: Path) -> None:
    """
    Line chart: mean PM2.5 by calendar month × station.
    Reveals winter peak due to temperature inversion — a key seasonal feature.
    """
    data = df[["station", "timestamp", "PM2.5"]].dropna().copy()
    data["month"] = data["timestamp"].dt.month
    agg  = data.groupby(["month", "station"])["PM2.5"].mean().reset_index()
    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for stn in STATIONS:
        s = agg[agg["station"] == stn].sort_values("month")
        ax.plot(s["month"], s["PM2.5"], marker="o", label=stn,
                color=STATION_COLORS[stn], linewidth=2.2, markersize=7)

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTHS)
    ymax = ax.get_ylim()[1]
    for span, label in [((0.5, 2.5), "Winter"), ((10.5, 12.5), "Winter"), ((5.5, 8.5), "Monsoon")]:
        fc = "#BBDEFB" if "Winter" in label else "#C8E6C9"
        ax.axvspan(*span, alpha=0.12, color=fc)
        ax.text(sum(span)/2, ymax * 0.97, label, ha="center", fontsize=9,
                color="#1565C0" if "Winter" in label else "#2E7D32", style="italic")

    ax.set_xlabel("Month")
    ax.set_ylabel("Mean PM2.5 (μg/m³)")
    ax.set_title(
        "Seasonal PM2.5 Pattern by Station\n"
        "Winter peaks (Jan–Feb, Nov–Dec) from temperature inversion; monsoon washout (Jun–Sep)"
    )
    ax.legend(title="Station", frameon=True)
    _save(fig, out)


def plot_pm25_station_comparison_bar(df: pd.DataFrame, out: Path) -> None:
    """
    Horizontal grouped bar: mean, median, 90th-percentile PM2.5 per station.
    Single-glance summary of relative pollution severity across stations.
    """
    data = df[["station", "PM2.5"]].dropna()
    stats = (data.groupby("station")["PM2.5"]
                 .agg(mean="mean", median="median", p90=lambda x: x.quantile(0.90))
                 .reindex(STATIONS))

    metrics  = ["mean", "median", "p90"]
    labels   = ["Mean", "Median", "90th Percentile"]
    m_colors = ["#1976D2", "#43A047", "#E53935"]
    x = np.arange(len(STATIONS))
    width = 0.26

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, label, color) in enumerate(zip(metrics, labels, m_colors)):
        vals = stats[metric].values
        bars = ax.bar(x + (i-1)*width, vals, width, label=label,
                      color=color, edgecolor="white", linewidth=0.8, alpha=0.9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(STATIONS, fontsize=13)
    ax.set_xlabel("Monitoring Station")
    ax.set_ylabel("PM2.5 (μg/m³)")
    ax.set_title(
        "PM2.5 Statistics Comparison Across Stations\n"
        "Mean / Median / 90th Percentile from full training dataset"
    )
    ax.legend(frameon=True)
    _save(fig, out)


# ─────────────────────────────────────────────────────────────────────────────
# 03  CORRELATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_per_station(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Separate lower-triangle Pearson correlation matrix for each station.
    Shows inter-pollutant relationships and what correlates most with PM2.5.
    """
    for stn in STATIONS:
        sdf  = df[df["station"] == stn].copy()
        avail = [c for c in CORE_COLS if c in sdf.columns and sdf[c].notna().sum() > 200]
        if len(avail) < 3:
            logging.warning("Station %s: too few valid core columns for correlation; skipping.", stn)
            continue
        corr = sdf[avail].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # hide upper triangle

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    annot=True, fmt=".2f", annot_kws={"size": 8.5},
                    square=True, linewidths=0.5, linecolor="white",
                    cbar_kws={"label": "Pearson r", "shrink": 0.75}, ax=ax)
        ax.set_title(
            f"Pearson Correlation Matrix — {stn} Station (Test Set)\n"
            "PM2.5 row reveals strongest co-pollutant predictors;  red = positive, blue = negative"
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
        _save(fig, out_dir / f"corr_matrix_{stn.lower()}.png")


def plot_predictor_correlation_bars(df: pd.DataFrame, out: Path) -> None:
    """
    2×2 subplot: bar chart of Pearson r between each core pollutant and PM2.5,
    per station.  Quickly shows which variables are the strongest predictors.
    """
    target = "PM2.5"
    preds  = [c for c in CORE_COLS if c != target]

    fig, axes = plt.subplots(2, 2, figsize=(17, 12), sharey=False)
    axes = axes.flatten()

    for idx, stn in enumerate(STATIONS):
        ax  = axes[idx]
        sdf = df[df["station"] == stn].copy()
        corr_vals = []
        names     = []
        for p in preds:
            if p in sdf.columns and sdf[p].notna().sum() > 200:
                r = float(sdf[[target, p]].corr().loc[target, p])
                corr_vals.append(r)
                names.append(p)

        order = np.argsort(corr_vals)[::-1]
        sorted_names = [names[i] for i in order]
        sorted_vals  = [corr_vals[i] for i in order]
        bar_colors   = ["#C62828" if v >= 0 else "#1565C0" for v in sorted_vals]

        ax.barh(sorted_names, sorted_vals, color=bar_colors, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.9, linestyle="--")
        ax.set_xlim(-1, 1)
        ax.set_title(stn, fontsize=13, fontweight="bold", color=STATION_COLORS[stn])
        ax.set_xlabel("Pearson r with PM2.5")
        ax.grid(True, axis="x", alpha=0.3)
        for name, v in zip(sorted_names, sorted_vals):
            ha = "left" if v >= 0 else "right"
            ax.text(v + (0.03 if v >= 0 else -0.03),
                    sorted_names.index(name), f"{v:+.2f}",
                    va="center", ha=ha, fontsize=8.5)

    fig.suptitle(
        "Correlation of Core Variables with PM2.5 — per Station (Test Set)\n"
        "Red bar = positive correlation, Blue = negative; longer bar = stronger predictor",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    _save(fig, out)


def plot_cross_station_corr(df: pd.DataFrame, out: Path) -> None:
    """
    Heatmap of Pearson r between simultaneous PM2.5 readings at all station pairs.
    High correlation implies shared regional sources and supports cross-station feature use.
    """
    pivot = (df[["station", "timestamp", "PM2.5"]]
             .dropna()
             .pivot_table(index="timestamp", columns="station", values="PM2.5", aggfunc="mean")
             .reindex(columns=STATIONS)
             .dropna(how="all"))
    corr = pivot.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, mask=mask, cmap="YlOrRd", vmin=0, vmax=1,
                annot=True, fmt=".3f", annot_kws={"size": 14, "fontweight": "bold"},
                square=True, linewidths=2.5, linecolor="white",
                cbar_kws={"label": "Pearson r (simultaneous PM2.5)", "shrink": 0.75}, ax=ax)
    ax.set_title(
        "Cross-Station PM2.5 Synchrony (Test Set)\n"
        "High r between stations → shared pollution episodes → cross-station lags used as features"
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    _save(fig, out)


# ─────────────────────────────────────────────────────────────────────────────
# 04  MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────

def plot_grouped_bars_per_metric(out_dir: Path) -> None:
    """
    Three separate grouped bar charts (RMSE, MAE, R²), one metric per plot.
    Each chart shows all 4 models for all 4 stations side-by-side.
    RF bars are capped so Ridge/XGB/LSTM differences remain readable.
    """
    df = pd.read_csv(METRICS_CSV)
    df = df[df["station"] != "ALL"].copy()
    df["model"] = df["model"].str.upper()

    all_models = ["RIDGE", "RF", "XGB", "LSTM"]
    x = np.arange(len(STATIONS))
    width = 0.2

    cfg = [
        ("rmse", "RMSE (μg/m³)",  "Lower = Better",  0, 12,   "RF RMSE far exceeds scale\n(RF unsuitable for this data)"),
        ("mae",  "MAE (μg/m³)",   "Lower = Better",  0,  8,   "RF MAE far exceeds scale"),
        ("r2",   "R² Score",      "Higher = Better", None, 1.05, None),
    ]

    for col, ylabel, note, ymin, ymax, annotation in cfg:
        fig, ax = plt.subplots(figsize=(14, 7))

        for i, model in enumerate(all_models):
            mdf  = df[df["model"] == model]
            vals = []
            for stn in STATIONS:
                row = mdf[mdf["station"] == stn]
                vals.append(float(row[col].iloc[0]) if not row.empty else np.nan)

            bars = ax.bar(x + (i - 1.5) * width, vals, width, label=model,
                          color=MODEL_COLORS[model], edgecolor="white",
                          linewidth=0.8, alpha=0.9)
            for bar, v in zip(bars, vals):
                if np.isnan(v):
                    continue
                # only annotate if within visible range
                if ymax is not None and v > ymax:
                    ax.text(bar.get_x() + bar.get_width()/2, ymax - 0.4,
                            f"{v:.1f}↑", ha="center", va="top", fontsize=7, color="white",
                            fontweight="bold")
                else:
                    offset = 0.07 if col != "r2" else 0.005
                    ax.text(bar.get_x() + bar.get_width()/2, v + offset,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(STATIONS, fontsize=13)
        ax.set_xlabel("Monitoring Station")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Model Comparison — {ylabel} per Station (Test Set)\n{note}")
        ax.legend(title="Model", frameon=True, loc="best")
        ax.grid(True, axis="y", alpha=0.35)

        if ymin is not None or ymax is not None:
            ax.set_ylim(ymin if ymin is not None else ax.get_ylim()[0], ymax)
        if col == "r2":
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        if annotation:
            ax.text(0.99, 0.98, annotation, transform=ax.transAxes,
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="#FFF9C4", alpha=0.9))

        _save(fig, out_dir / f"metric_{col}_grouped.png")


def plot_performance_heatmap(out: Path) -> None:
    """
    Side-by-side heatmaps: RMSE and R² for all 4 models × 4 stations.
    Dual view makes it easy to spot which model dominates where.
    """
    df = pd.read_csv(METRICS_CSV)
    df = df[df["station"] != "ALL"].copy()
    df["model"] = df["model"].str.upper()

    col_order = ["RIDGE", "XGB", "LSTM", "RF"]
    piv_rmse = df.pivot(index="station", columns="model", values="rmse").reindex(
        index=STATIONS, columns=col_order)
    piv_r2   = df.pivot(index="station", columns="model", values="r2").reindex(
        index=STATIONS, columns=col_order)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))

    # RMSE — cap RF colour at 15 for readability; true values shown as text
    rmse_clipped = piv_rmse.clip(upper=15)
    sns.heatmap(rmse_clipped, ax=axes[0], cmap="YlOrRd",
                annot=piv_rmse.round(2), fmt="",
                annot_kws={"size": 12, "fontweight": "bold"},
                linewidths=1.5, linecolor="white",
                cbar_kws={"label": "RMSE (μg/m³) — RF colour capped at 15"})
    axes[0].set_title("RMSE per Station × Model\n(Lower = Better; RF colour capped)", fontsize=12)
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("Station")
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)

    sns.heatmap(piv_r2, ax=axes[1], cmap="RdYlGn", vmin=-2.5, vmax=1.0,
                annot=True, fmt=".3f",
                annot_kws={"size": 12, "fontweight": "bold"},
                linewidths=1.5, linecolor="white",
                cbar_kws={"label": "R² Score (Higher = Better, max=1.0)"})
    axes[1].set_title("R² per Station × Model\n(Higher = Better; RF is negative = worse than baseline mean)", fontsize=12)
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("Station")
    axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)

    fig.suptitle("Complete Model Performance Summary — Test Set  (RF excluded from routing)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, out)


def plot_all_models_comparison(out: Path) -> None:
    """
    RMSE and R² side-by-side for Ridge / XGB / LSTM / Routed (best classical).
    Routed = per-station best classical model (XGB for AIIMS, Ridge elsewhere).
    """
    df = pd.read_csv(METRICS_CSV)
    df = df[df["station"] != "ALL"].copy()
    df["model"] = df["model"].str.upper()

    routed_json = json.loads(JSON_ROUTED.read_text(encoding="utf-8"))
    routing = routed_json["station_routing"]
    routed_rows = [
        {"station": stn,
         "model": "ROUTED",
         "rmse": routing[stn]["test_metrics"]["rmse"],
         "mae":  routing[stn]["test_metrics"]["mae"],
         "r2":   routing[stn]["test_metrics"]["r2"]}
        for stn in STATIONS
    ]
    df_all = pd.concat([df, pd.DataFrame(routed_rows)], ignore_index=True)

    show = ["RIDGE", "XGB", "LSTM", "ROUTED"]
    x     = np.arange(len(STATIONS))
    width = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, (col, ylabel, ylim, note) in zip(axes, [
        ("rmse", "RMSE (μg/m³)", (0, 11),   "Lower = Better"),
        ("r2",   "R² Score",     (0.82, 1.02), "Higher = Better"),
    ]):
        for i, model in enumerate(show):
            mdf  = df_all[df_all["model"] == model]
            vals = [float(mdf[mdf["station"] == stn][col].iloc[0])
                    if not mdf[mdf["station"] == stn].empty else np.nan
                    for stn in STATIONS]
            bars = ax.bar(x + (i-1.5)*width, vals, width, label=model,
                          color=MODEL_COLORS[model], edgecolor="white", linewidth=0.8, alpha=0.9)
            for bar, v in zip(bars, vals):
                if not np.isnan(v) and ylim[0] <= v <= ylim[1]:
                    offset = 0.08 if col == "rmse" else 0.003
                    ax.text(bar.get_x() + bar.get_width()/2, v + offset,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(STATIONS, fontsize=12)
        ax.set_xlabel("Station")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_title(f"{ylabel} — Ridge vs XGB vs LSTM vs Routed\n{note}")
        ax.legend(title="Model", frameon=True, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Ridge vs XGBoost vs LSTM vs Routed Strategy — Test Set\n"
        "ROUTED = best classical model chosen per station (validation RMSE criterion)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    _save(fig, out)


def plot_feature_counts(out: Path) -> None:
    """
    Bar chart of #features used per station model after lag/rolling engineering.
    Explains why BHATAGAON has most features (cross-station SILTARA columns included).
    """
    ps = json.loads(JSON_PER_STATION.read_text(encoding="utf-8"))["per_station"]
    n_feats = [ps[stn]["n_features"] for stn in STATIONS]
    lstm_feats = 498  # from lstm_metrics.json

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [STATION_COLORS[stn] for stn in STATIONS]
    bars   = ax.bar(STATIONS, n_feats, color=colors, edgecolor="white", linewidth=0.8, alpha=0.9)
    for bar, v in zip(bars, n_feats):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
                f"{v}", ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.axhline(lstm_feats, color=MODEL_COLORS["LSTM"], linewidth=2,
               linestyle="--", label=f"LSTM feature set ({lstm_feats} after dropping leaky cols)")
    ax.set_xlabel("Station")
    ax.set_ylabel("Number of Feature Columns")
    ax.set_title(
        "Feature Count per Station After Lag & Rolling Window Engineering\n"
        "BHATAGAON: 483 features — includes cross-station SILTARA lags (+224 vs AIIMS/IGKV)"
    )
    ax.legend(frameon=True, fontsize=10)
    ax.set_ylim(0, 620)
    ax.grid(True, axis="y", alpha=0.35)

    # Annotate cross-station bonus
    ax.annotate(
        "Cross-station\nSILTARA features",
        xy=(1, ps["BHATAGAON"]["n_features"]), xytext=(1.7, 555),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color="#333"),
        fontsize=9, ha="center"
    )
    _save(fig, out)


def plot_summary_scorecard(out: Path) -> None:
    """Formatted table figure — single-slide summary for the viva."""
    data = {
        "Station":              STATIONS,
        "Routing Decision":     ["XGBoost", "Ridge", "Ridge", "Ridge"],
        "Routed RMSE":          [2.67, 5.15, 2.02, 4.82],
        "Routed MAE":           [1.31, 1.62, 1.08, 1.68],
        "Routed R²":            [0.9833, 0.9111, 0.9835, 0.9598],
        "LSTM RMSE":            [5.90, 6.73, 3.21, 8.21],
        "LSTM R²":              [0.9183, 0.8302, 0.9586, 0.8842],
        "# Features":           [259, 483, 259, 274],
        "# Test Rows":          [18027, 18289, 14294, 14452],
    }
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(19, 4))
    ax.axis("off")

    # Format numerics
    cell_text = []
    for _, row in df.iterrows():
        row_display = []
        for val in row:
            if isinstance(val, float):
                row_display.append(f"{val:.4f}" if val < 10 else f"{val:.2f}")
            elif isinstance(val, int):
                row_display.append(f"{val:,}")
            else:
                row_display.append(str(val))
        cell_text.append(row_display)

    tbl = ax.table(cellText=cell_text, colLabels=df.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1, 2.4)

    header_color = "#1565C0"
    for j in range(len(df.columns)):
        tbl[0, j].set_facecolor(header_color)
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    row_colors = {
        "AIIMS":     "#E3F2FD",
        "BHATAGAON": "#E8F5E9",
        "IGKV":      "#FFF3E0",
        "SILTARA":   "#FCE4EC",
    }
    for i, stn in enumerate(STATIONS, start=1):
        for j in range(len(df.columns)):
            tbl[i, j].set_facecolor(row_colors[stn])
            if j in (2, 4):  # Routed RMSE, Routed R² — bold
                tbl[i, j].set_text_props(fontweight="bold")

    ax.set_title(
        "Final Results Scorecard — ML-Based PM2.5 Prediction\n"
        "Target: PM2.5 (μg/m³) | 15-min resolution | Chronological 70/15/15 split | Raipur, Chhattisgarh",
        fontsize=12, fontweight="bold", pad=20
    )
    _save(fig, out)


# ─────────────────────────────────────────────────────────────────────────────
# 05  LSTM
# ─────────────────────────────────────────────────────────────────────────────

def plot_lstm_history(out: Path) -> None:
    """
    Left: train vs validation loss over 14 epochs.
    Right: learning-rate schedule (ReduceLROnPlateau).
    Both panels contextualize LSTM convergence behaviour.
    """
    lstm = json.loads(JSON_LSTM.read_text(encoding="utf-8"))
    hist = lstm["history"]
    epochs = list(range(1, len(hist["loss"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Loss curves
    ax = axes[0]
    ax.plot(epochs, hist["loss"],     "o-", color="#1976D2", label="Train Loss",      linewidth=2.2, markersize=7)
    ax.plot(epochs, hist["val_loss"], "s--", color="#E53935", label="Validation Loss", linewidth=2.2, markersize=7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (normalised scale)")
    ax.set_title("LSTM Training & Validation Loss\n14 epochs with ReduceLROnPlateau — val_loss converges ~0.213")
    ax.legend(frameon=True)
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.35)
    ax.annotate(f"Final train: {hist['loss'][-1]:.4f}",
                xy=(epochs[-1], hist["loss"][-1]),
                xytext=(-60, 15), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="#1976D2"), color="#1976D2", fontsize=9)
    ax.annotate(f"Final val: {hist['val_loss'][-1]:.4f}",
                xy=(epochs[-1], hist["val_loss"][-1]),
                xytext=(-80, 15), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="#E53935"), color="#E53935", fontsize=9)

    # LR schedule
    ax2 = axes[1]
    ax2.step(epochs, hist["learning_rate"], where="post", color="#7B1FA2", linewidth=2.2)
    ax2.fill_between(epochs, hist["learning_rate"], step="post", alpha=0.2, color="#7B1FA2")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule (ReduceLROnPlateau)\nHalved when val_loss stagnates; starts at 1e-3")
    ax2.set_xticks(epochs)
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax2.grid(True, alpha=0.35)

    plt.tight_layout()
    _save(fig, out)


def plot_lstm_per_station(out: Path) -> None:
    """
    3-panel bar chart (RMSE, MAE, R²) for LSTM per station.
    Shows spatial heterogeneity in LSTM performance across monitoring sites.
    """
    lstm      = json.loads(JSON_LSTM.read_text(encoding="utf-8"))
    by_stn    = lstm["test_metrics_by_station"]
    colors    = [STATION_COLORS[s] for s in STATIONS]

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    for ax, metric, label in zip(axes,
                                  ["rmse", "mae", "r2"],
                                  ["RMSE (μg/m³)", "MAE (μg/m³)", "R² Score"]):
        vals = [by_stn[stn][metric] for stn in STATIONS]
        bars = ax.bar(STATIONS, vals, color=colors, edgecolor="white", linewidth=0.8, alpha=0.9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_title(f"LSTM — {label}", fontsize=12)
        ax.set_xlabel("Station")
        ax.set_ylabel(label)
        ax.grid(True, axis="y", alpha=0.35)
        if metric == "r2":
            ax.set_ylim(0.78, 1.02)
            ax.axhline(0.9, color="#2E7D32", linewidth=1.5, linestyle="--", alpha=0.7, label="R²=0.90")
            ax.legend(fontsize=9)

    fig.suptitle(
        "LSTM Test Performance per Station\n"
        "IGKV best (R²=0.959) | SILTARA hardest (RMSE=8.21) | All R² > 0.83",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    _save(fig, out)


# ─────────────────────────────────────────────────────────────────────────────
# 06  ROUTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_routing_val_heatmap(out: Path) -> None:
    """
    Heatmap of validation RMSE for every (station, model) pair.
    Blue border highlights the chosen model per station.
    Demonstrates why RF was discarded and shows Ridge vs XGB competition.
    """
    routed  = json.loads(JSON_ROUTED.read_text(encoding="utf-8"))["station_routing"]
    models  = ["ridge", "rf", "xgb"]

    raw = {stn: {m: routed[stn]["val_scores"].get(m, {}).get("rmse", np.nan)
                 for m in models}
           for stn in STATIONS}
    df  = pd.DataFrame(raw).T.reindex(index=STATIONS)
    df.columns = [m.upper() for m in models]

    chosen_by_station = {stn: routed[stn]["chosen_model"].upper() for stn in STATIONS}

    fig, ax = plt.subplots(figsize=(10, 6))
    # Cap RF colour at 30 for readability; true value shown as text
    df_clipped = df.copy().clip(upper=30)
    sns.heatmap(df_clipped, cmap="YlOrRd_r", ax=ax,
                annot=df.round(2), fmt="",
                annot_kws={"size": 13, "fontweight": "bold"},
                linewidths=2, linecolor="white",
                cbar_kws={"label": "Validation RMSE (μg/m³) — RF colour capped at 30"})

    col_map = {"RIDGE": 0, "RF": 1, "XGB": 2}
    for i, stn in enumerate(STATIONS):
        j = col_map[chosen_by_station[stn]]
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="#1565C0", lw=3.5))
        ax.text(j + 0.5, i + 0.17, "★ CHOSEN", ha="center", va="top",
                fontsize=8.5, color="#1565C0", fontweight="bold")

    ax.set_title(
        "Model Routing — Validation RMSE per Station × Classical Model\n"
        "★ CHOSEN = model selected for final test (lowest val RMSE) | RF consistently worst",
        fontsize=11
    )
    ax.set_xlabel("Classical Model")
    ax.set_ylabel("Station")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    _save(fig, out)


def plot_routing_summary(out: Path) -> None:
    """
    Left: val RMSE vs test RMSE for the chosen model per station (calibration check).
    Right: test R² of routed model per station (final performance).
    """
    routed  = json.loads(JSON_ROUTED.read_text(encoding="utf-8"))["station_routing"]
    chosen  = [routed[stn]["chosen_model"].upper() for stn in STATIONS]
    val_r   = [routed[stn]["val_scores"][routed[stn]["chosen_model"]]["rmse"] for stn in STATIONS]
    test_r  = [routed[stn]["test_metrics"]["rmse"]  for stn in STATIONS]
    test_r2 = [routed[stn]["test_metrics"]["r2"]    for stn in STATIONS]

    x = np.arange(len(STATIONS))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: val vs test RMSE
    ax = axes[0]
    ax.bar(x - w/2, val_r,  w, label="Val RMSE (routing criterion)", color="#FFA726", edgecolor="white")
    ax.bar(x + w/2, test_r, w, label="Test RMSE (final evaluation)", color="#1E88E5", edgecolor="white")
    for i, (v, t) in enumerate(zip(val_r, test_r)):
        ax.text(x[i]-w/2, v+0.2, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(x[i]+w/2, t+0.2, f"{t:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{stn}\n({m})" for stn, m in zip(STATIONS, chosen)], fontsize=11)
    ax.set_ylabel("RMSE (μg/m³)")
    ax.set_title("Routed Model: Validation vs Test RMSE\n(Station label shows chosen model)")
    ax.legend(frameon=True)
    ax.grid(True, axis="y", alpha=0.35)

    # Right: test R²
    ax2 = axes[1]
    colors = [STATION_COLORS[stn] for stn in STATIONS]
    bars   = ax2.bar(x, test_r2, color=colors, edgecolor="white", linewidth=0.8, alpha=0.9)
    for bar, v, m in zip(bars, test_r2, chosen):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                 f"R²={v:.4f}\n({m})", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(STATIONS, fontsize=12)
    ax2.set_ylabel("R² Score")
    ax2.set_ylim(0.88, 1.01)
    ax2.axhline(0.95, color="#C62828", linewidth=1.5, linestyle="--", alpha=0.7, label="R²=0.95 reference")
    ax2.set_title("Routed Model Test R² per Station\n(AIIMS & IGKV exceed 0.983)")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.35)

    fig.suptitle(
        "Model Routing Results — Best Classical Model per Station\n"
        "Criterion: lowest validation RMSE among (Ridge, RF, XGBoost)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    _save(fig, out)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    _setup()

    logging.info("Loading parquet files...")
    df_train = _load_parquet(PARQUET_TRAIN)   # full — for temporal/distributional plots
    df_test  = _load_parquet(PARQUET_TEST)    # full — for correlation + cross-station

    # Sample test parquet for heavy pairwise correlation (keeps memory sane)
    df_test_corr = (df_test.sample(n=min(60_000, len(df_test)), random_state=42)
                    if len(df_test) > 60_000 else df_test)

    # ── 01 Data Overview ──────────────────────────────────────────────────────
    logging.info("01 Data Overview")
    d = OUT_DIR / "01_data_overview"
    plot_dataset_size(d / "01_dataset_size_by_station.png")
    plot_pm25_violin(df_train, d / "02_pm25_distribution_violin_box.png")
    plot_pm25_by_year(df_train, d / "03_pm25_by_year_boxplot.png")
    plot_pm25_station_comparison_bar(df_train, d / "04_pm25_mean_median_p90_bar.png")

    # ── 02 Inter-Station ─────────────────────────────────────────────────────
    logging.info("02 Inter-Station Patterns")
    d = OUT_DIR / "02_inter_station"
    plot_hourly_heatmap(df_train, d / "01_pm25_diurnal_heatmap.png")
    plot_dow_pattern(df_train,    d / "02_pm25_weekly_pattern.png")
    plot_monthly_pattern(df_train, d / "03_pm25_seasonal_pattern.png")

    # ── 03 Correlation ───────────────────────────────────────────────────────
    logging.info("03 Correlation")
    d = OUT_DIR / "03_correlation"
    plot_correlation_per_station(df_test_corr, d)          # 4 files: one per station
    plot_predictor_correlation_bars(df_test_corr, d / "predictor_r_with_pm25_2x2.png")
    plot_cross_station_corr(df_test, d / "cross_station_pm25_synchrony.png")

    # ── 04 Model Performance ─────────────────────────────────────────────────
    logging.info("04 Model Performance")
    d = OUT_DIR / "04_model_performance"
    plot_grouped_bars_per_metric(d)                        # 3 files: rmse, mae, r2
    plot_performance_heatmap(d / "rmse_r2_heatmap.png")
    plot_all_models_comparison(d / "ridge_xgb_lstm_routed_comparison.png")
    plot_feature_counts(d / "feature_count_per_station.png")
    plot_summary_scorecard(d / "final_scorecard.png")

    # ── 05 LSTM ──────────────────────────────────────────────────────────────
    logging.info("05 LSTM")
    d = OUT_DIR / "05_lstm"
    plot_lstm_history(d / "lstm_training_history.png")
    plot_lstm_per_station(d / "lstm_per_station_performance.png")

    # ── 06 Routing ───────────────────────────────────────────────────────────
    logging.info("06 Routing")
    d = OUT_DIR / "06_routing"
    plot_routing_val_heatmap(d / "routing_val_rmse_heatmap.png")
    plot_routing_summary(d / "routing_val_vs_test_summary.png")

    logging.info("All done — plots in: %s", OUT_DIR)


if __name__ == "__main__":
    main()
