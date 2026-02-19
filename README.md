# 📈 HW_forecast — Holt-Winters Rolling Forecast Dashboard

**Author:** JC — Jose Sirias  
**Framework:** Streamlit · Python  
**Methodology:** CRISP-DM  

A production-grade Streamlit dashboard for time series forecasting using Holt-Winters (Triple Exponential Smoothing) with rolling-origin backtesting, bootstrap confidence intervals, and automated deployment scoring.

---

## 1. Business Understanding

### Objective
Forecast daily USD transaction volumes to support operational planning, budget allocation, and capacity management.

### Key Questions
- What will daily approved transaction volume look like over the next 7–30 days?
- How accurate and stable is the model across different time periods?
- Is the forecast reliable enough for production deployment?

### Success Criteria
- sMAPE < 20% across rolling backtest windows
- MASE < 1.5 (must beat seasonal naive baseline)
- CI95 coverage > 70% (confidence intervals are calibrated)
- Consistent performance across time (CV stability < 30%)

---

## 2. Data Understanding

### Input
- **Format:** CSV file uploaded via the dashboard
- **Required columns:** `date`, `usd`, `status`
- **Filtering:** Only rows with `status == 'Approved'` are used
- **Aggregation:** Daily sum of `usd` by `date`

### Preprocessing
- Datetime index with daily frequency (`asfreq('D')`)
- Missing dates filled via linear interpolation
- Incomplete current month excluded from backtesting (used for forecast vs actual comparison)

### Exploratory Analysis (Tab 1)
- Time series visualization with train/val/test split overlay
- ADF stationarity test
- Seasonal decomposition (additive or multiplicative)
- Auto-detection of seasonal type via STL amplitude-level correlation

---

## 3. Data Preparation

### Train / Validation / Test Split
- Configurable split ratio (default: 70/15/15)
- Alternative 90/5/5 for maximum training data
- Side-by-side split comparison available (Tab 5)
- Guard: test set must be ≥ 2% of data

### Seasonal Period
- Default: 7 (weekly seasonality for daily data)
- Configurable via sidebar

---

## 4. Modeling

### Algorithm
**Holt-Winters (Triple Exponential Smoothing)** via `statsmodels.tsa.holtwinters.ExponentialSmoothing`

### Parameters (all configurable via sidebar)
| Parameter | Description | Default |
|-----------|-------------|---------|
| α (alpha) | Level smoothing | 0.4 |
| β (beta) | Trend smoothing | 0.05 |
| γ (gamma) | Seasonal smoothing | 0.3 |
| φ (phi) | Damping factor | 0.98 |
| Trend | Additive trend | `add` |
| Seasonal | Auto-detected (STL) | `mul` or `add` |
| Damped | Damped trend | Yes |
| Optimized | Let statsmodels optimize | No |

### Confidence Intervals
- Bootstrap residual resampling (500 paths for backtest, 1000 for forecast)
- 90% and 95% prediction intervals
- Non-negative floor (`np.maximum(fc, 0)`)

---

## 5. Evaluation

### Rolling-Origin Backtest (Tab 2)
- Non-overlapping sliding windows across the test set
- Per-window metrics: sMAPE, MASE, MAPE, RMSE, Bias, CI95 coverage
- Accuracy decay analysis by forecast day

### Residual Diagnostics (Tab 3)
- **Ljung-Box test** — autocorrelation in residuals (concatenated + per-window)
- **Shapiro-Wilk test** — normality of residuals
- **Bias analysis** — systematic over/under-prediction (< 5% threshold)
- **Lag-1 ACF** — per-window independence check
- **ACF plot** — residual autocorrelation structure
- **Residual histogram & time plot**

### CV Stability (Tab 3)
- Expanding-window cross-validation on training data
- sMAPE coefficient of variation across folds
- Stability threshold: CV < 30%

### Forecast vs Actual (Tab 4)
- Automatic overlay of current month actuals against forecast
- Per-day comparison table with CI95 coverage check
- Cumulative actual vs forecast totals

### Deployment Scorecard (Tab 4)
Automated go/no-go decision with up to 8 criteria:

| Criterion | Threshold |
|-----------|-----------|
| sMAPE | < 20% |
| MASE | < 1.5 |
| R² | > 0 |
| Window CV | < 50% |
| CI95 Coverage | > 70% |
| Ljung-Box (per-window) | Majority pass |
| Bias | < 5% |
| CV Stability | < 30% |

**Verdict scale:** 🟢 Production (≥80%) · 🟡 Conditional (≥60%) · 🟠 Limited (≥40%) · 🔴 Not Ready (<40%)

### Split Comparison (Tab 5)
- Side-by-side 70/15/15 vs 90/5/5 with h=7 and h=14
- Comparative bar charts (sMAPE, MASE, R²)
- Automated winner selection

---

## 6. Deployment

### Local
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Streamlit Community Cloud
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository → select `streamlit_app.py` → Deploy

### Configuration
`.streamlit/config.toml`:
- Max upload size: 500 MB
- Session timeout: 2 hours (7200s)

---

## Project Structure

```
HW_forecast/
├── .gitignore
├── .streamlit/
│   └── config.toml
├── README.md
├── requirements.txt
└── streamlit_app.py
```

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Dashboard | Streamlit |
| Forecasting | statsmodels (ExponentialSmoothing) |
| Statistics | scipy (Shapiro-Wilk), statsmodels (ADF, Ljung-Box, ACF, STL) |
| Metrics | scikit-learn (MAE, MSE, R²) |
| Visualization | Plotly |
| Data | pandas, numpy |

---

## License

Private project — Promidat · JC Jose Sirias
