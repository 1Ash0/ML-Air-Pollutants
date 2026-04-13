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

