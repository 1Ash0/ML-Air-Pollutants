# Final Results Summary (v4)

Target: `PM2.5`

## Best Final System (Recommended)

**Per-station routed classical ensemble** (chosen by validation RMSE per station):
- AIIMS: **XGBoost**
- BHATAGAON: **Ridge**
- IGKV: **Ridge**
- SILTARA: **Ridge**

Test (global):
- RMSE: **3.94**
- MAE: **1.43**
- R²: **0.9607**

Source: `artifacts/classical_metrics_routed_v4.json`

## LSTM (Deep Learning baseline)

Test (global):
- RMSE: **6.29**
- MAE: **3.04**
- R²: **0.8978**

Source: `artifacts/lstm_metrics.json`

## Unified Comparison Table + Plots

- Table: `artifacts/metrics.csv`
- Plots: `artifacts/plots/*.png`

## Multi-Pollutant Forecasting (t+15 min) — Extension

To address class feedback, we also train a **single multi-output model** that forecasts multiple pollutants simultaneously (default horizon: **+15 minutes**).

- Metrics: `artifacts/multioutput_metrics.csv`
- Metadata: `artifacts/multioutput_meta.json`
- Plots: `artifacts/plots/viva/07_multioutput/*`

## Comparative Study: Global vs Per-Station vs Routed

To address the “per-station overfitting” viva question, train global models and compare:

- Output: `artifacts/pm25_comparison.json`
- Script: `8_compare_global_vs_station.py`
