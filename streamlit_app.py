# ==============================================================================
# Holt-Winters Rolling Forecast — Streamlit MVP
# ==============================================================================
# CRISP-DM: Full pipeline from data loading to production forecast
# Based on test_17 (rolling engine) + test_18 (split comparison)
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import warnings
import time
import json
import os
import io

warnings.filterwarnings('ignore')

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="HW Forecast Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# METRIC FUNCTIONS
# ==============================================================================
def smape(actual, predicted):
    a, p = np.array(actual, dtype=float), np.array(predicted, dtype=float)
    denom = (np.abs(a) + np.abs(p))
    mask = denom > 0
    if mask.sum() == 0:
        return 0.0
    return float(100 * np.mean(2 * np.abs(a[mask] - p[mask]) / denom[mask]))

def mase(actual, predicted, train_data, m=7):
    a, p = np.array(actual, dtype=float), np.array(predicted, dtype=float)
    t = np.array(train_data, dtype=float)
    naive_errors = np.abs(t[m:] - t[:-m])
    d = np.mean(naive_errors)
    if d == 0 or np.isnan(d):
        return 999.0  # safe sentinel instead of inf (avoids formatting crashes)
    return float(np.mean(np.abs(a - p)) / d)

def compute_all_metrics(actual, predicted, train_data, m=7):
    a, p = np.array(actual), np.array(predicted)
    sm = smape(a, p)
    ms = mase(a, p, train_data, m)
    rm = np.sqrt(mean_squared_error(a, p))
    ma = mean_absolute_error(a, p)
    r2 = r2_score(a, p) if len(a) > 1 else 0
    mp = 100 * np.mean(np.abs((a - p) / np.where(a == 0, 1, a)))
    bias = np.mean(p - a)
    return {'sMAPE': sm, 'MASE': ms, 'RMSE': rm, 'MAE': ma,
            'R²': r2, 'MAPE': mp, 'Bias': bias}

# ==============================================================================
# DATA LOADING (cached)
# ==============================================================================
@st.cache_data
def load_data(csv_bytes: bytes):
    """Load and prepare the volume data from raw CSV bytes."""
    df_raw = pd.read_csv(io.BytesIO(csv_bytes))
    df = df_raw[df_raw['status'] == 'Approved'].groupby('date')['usd'].sum().reset_index()
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])
    series = df.set_index('ds')['y'].asfreq('D')
    nan_count = series.isna().sum()
    if nan_count > 0:
        series = series.interpolate(method='linear')
    return series, nan_count, len(df_raw)

@st.cache_data
def create_split(series_data, train_pct, val_pct, test_pct):
    """Create train/val/test split."""
    n = len(series_data)
    train_end = int(n * train_pct / 100)
    val_end = int(n * (train_pct + val_pct) / 100)
    train = series_data.iloc[:train_end]
    val = series_data.iloc[train_end:val_end]
    test = series_data.iloc[val_end:]
    return train, val, test, train_end, val_end

@st.cache_data
def run_adf_test(series_values):
    """Run ADF test on series."""
    result = adfuller(series_values, autolag='AIC')
    return {
        'statistic': result[0],
        'p_value': result[1],
        'lags': result[2],
        'critical_values': result[4]
    }

@st.cache_data
def run_decomposition(series_values, _series_index, model, period):
    """Run seasonal decomposition."""
    s = pd.Series(series_values, index=_series_index)
    decomp = seasonal_decompose(s, model=model, period=period)
    return decomp.trend, decomp.seasonal, decomp.resid

@st.cache_data
def detect_seasonal_type(train_values, period=7):
    """Auto-detect mul vs add seasonal via STL amplitude-level correlation."""
    train_s = pd.Series(train_values)
    stl = STL(train_s, period=period, robust=True).fit()
    seasonal_comp = stl.seasonal
    trend_comp = stl.trend
    mask = ~np.isnan(trend_comp)
    if mask.sum() < 10:
        return 'add', 0.0, 0.0
    corr = np.corrcoef(np.abs(seasonal_comp[mask]), trend_comp[mask])[0, 1]
    strength = 1 - (np.var(stl.resid) / np.var(stl.resid + stl.seasonal))
    rec = 'mul' if abs(corr) > 0.3 else 'add'
    return rec, float(corr), float(strength)

@st.cache_data
def run_residual_diagnostics(actual_values, forecast_values):
    """Ljung-Box, Shapiro-Wilk, bias analysis on residuals."""
    residuals = np.array(actual_values) - np.array(forecast_values)
    # Ljung-Box
    lb = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
    lb_pass = bool((lb['lb_pvalue'] > 0.05).all())
    lb_pvals = lb['lb_pvalue'].values.tolist()
    # Shapiro-Wilk (max 5000 samples)
    r_sample = residuals[:5000] if len(residuals) > 5000 else residuals
    sw_stat, sw_p = shapiro(r_sample)
    sw_pass = bool(sw_p > 0.05)
    # Bias
    mean_resid = float(np.mean(residuals))
    mean_actual = float(np.mean(actual_values))
    bias_pct = abs(mean_resid / mean_actual * 100) if mean_actual != 0 else 0
    bias_pass = bias_pct < 5
    # ACF values for plotting
    from statsmodels.tsa.stattools import acf
    acf_vals = acf(residuals, nlags=min(30, len(residuals) - 1), fft=True)
    return {
        'lb_pass': lb_pass, 'lb_pvals': lb_pvals,
        'sw_pass': sw_pass, 'sw_stat': float(sw_stat), 'sw_p': float(sw_p),
        'bias': mean_resid, 'bias_pct': float(bias_pct), 'bias_pass': bias_pass,
        'residuals': residuals.tolist(),
        'acf_vals': acf_vals.tolist(),
        'std': float(np.std(residuals)),
        'skew': float(pd.Series(residuals).skew()),
        'kurtosis': float(pd.Series(residuals).kurtosis())
    }

@st.cache_data
def run_cv_stability(train_values, _train_index, n_splits,
                    alpha, beta, gamma, phi,
                    trend, seasonal, seasonal_periods, damped, optimized,
                    horizon=7):
    """Expanding-window CV stability check on train data."""
    train_s = pd.Series(train_values, index=_train_index).asfreq('D')
    fold_size = len(train_s) // (n_splits + 1)
    fold_results = []
    for fold_idx in range(n_splits):
        tr_end = (fold_idx + 2) * fold_size
        va_start = tr_end
        va_end = min(va_start + fold_size, len(train_s))
        if va_end <= va_start:
            continue
        fold_train = train_s.iloc[:tr_end]
        fold_val = train_s.iloc[va_start:va_end]
        try:
            mdl = ExponentialSmoothing(
                fold_train, trend=trend, seasonal=seasonal,
                seasonal_periods=seasonal_periods, damped_trend=damped
            )
            fit_kw = {}
            if optimized:
                fit_kw['optimized'] = True
            else:
                fit_kw['optimized'] = False
                fit_kw['smoothing_level'] = alpha
                fit_kw['smoothing_trend'] = beta
                fit_kw['smoothing_seasonal'] = gamma
                fit_kw['damping_trend'] = phi
            fit = mdl.fit(**fit_kw)
            fc = np.maximum(fit.forecast(len(fold_val)), 0)
            sm = float(smape(fold_val.values, fc.values))
            ms = float(mase(fold_val.values, fc.values, fold_train.values, m=7))
            fold_results.append({
                'fold': fold_idx + 1,
                'train_days': len(fold_train),
                'val_days': len(fold_val),
                'smape': sm, 'mase': ms,
                'status': 'OK'
            })
        except Exception as e:
            fold_results.append({
                'fold': fold_idx + 1,
                'train_days': tr_end,
                'val_days': va_end - va_start,
                'smape': np.nan, 'mase': np.nan,
                'status': f'FAIL: {str(e)[:60]}'
            })
    return fold_results

@st.cache_data
def run_per_window_diagnostics(windows_json):
    """Per-window Ljung-Box and lag-1 ACF.
    Tests autocorrelation WITHIN each rolling window independently."""
    results = []
    for i, w in enumerate(windows_json):
        if w.get('status') != 'OK':
            continue
        actual = np.array(w['actual'])
        forecast = np.array(w['forecast'])
        residuals = actual - forecast
        n = len(residuals)

        # Lag-1 autocorrelation
        if n > 2:
            lag1_acf = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
        else:
            lag1_acf = 0.0
        ci_bound = 1.96 / np.sqrt(n)
        lag1_pass = abs(lag1_acf) < ci_bound

        # Ljung-Box with appropriate lags (max = n - 2)
        max_lag = min(3, n - 2)
        if max_lag >= 1:
            lb = acorr_ljungbox(residuals, lags=list(range(1, max_lag + 1)), return_df=True)
            lb_min_p = float(lb['lb_pvalue'].min())
            lb_pass = lb_min_p > 0.05
        else:
            lb_min_p = 1.0
            lb_pass = True

        results.append({
            'Window': i + 1,
            'Period': f"{w['forecast_start'][:10]} → {w['forecast_end'][:10]}",
            'Lag-1 ACF': round(lag1_acf, 4),
            'ACF Pass': lag1_pass,
            'LB min-p': round(lb_min_p, 4),
            'LB Pass': lb_pass,
            'n': n
        })
    return results

# ==============================================================================
# ROLLING FORECAST ENGINE (cached)
# ==============================================================================
@st.cache_data
def hw_rolling_backtest(series_values, _series_index, test_start_idx, horizon,
                        alpha, beta, gamma, phi,
                        trend, seasonal, seasonal_periods, damped, optimized,
                        seed=42):
    """Run rolling-origin backtest."""
    series_data = pd.Series(series_values, index=_series_index).asfreq('D')
    rng = np.random.default_rng(seed)
    windows = []
    origin = test_start_idx

    while origin + horizon <= len(series_data):
        train_data = series_data.iloc[:origin]
        actual_data = series_data.iloc[origin:origin + horizon]

        try:
            mdl = ExponentialSmoothing(
                train_data,
                trend=trend, seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                damped_trend=damped
            )
            fit_kw = {}
            if optimized:
                fit_kw['optimized'] = True
            else:
                fit_kw['optimized'] = False
                fit_kw['smoothing_level'] = alpha
                fit_kw['smoothing_trend'] = beta
                fit_kw['smoothing_seasonal'] = gamma
                fit_kw['damping_trend'] = phi

            fit = mdl.fit(**fit_kw)
            fc = np.maximum(fit.forecast(horizon), 0)

            # Bootstrap CI
            resid = train_data.values - fit.fittedvalues.values
            valid_r = resid[~np.isnan(resid)]
            boot = np.zeros((500, horizon))
            for b in range(500):
                boot[b, :] = fc.values + rng.choice(valid_r, size=horizon, replace=True)
            lo_95 = np.percentile(boot, 2.5, axis=0)
            hi_95 = np.percentile(boot, 97.5, axis=0)

            w = {
                'origin_idx': origin,
                'forecast_start': str(actual_data.index[0]),
                'forecast_end': str(actual_data.index[-1]),
                'actual': actual_data.values.tolist(),
                'forecast': fc.values.tolist(),
                'dates': [str(d) for d in actual_data.index],
                'lo_95': lo_95.tolist(),
                'hi_95': hi_95.tolist(),
                'smape': float(smape(actual_data.values, fc.values)),
                'mase': float(mase(actual_data.values, fc.values, train_data.values, m=7)),
                'mape': float(100 * np.mean(np.abs((actual_data.values - fc.values) / actual_data.values))),
                'rmse': float(np.sqrt(mean_squared_error(actual_data.values, fc.values))),
                'bias': float(np.mean(fc.values - actual_data.values)),
                'ci95_coverage': float(np.mean((actual_data.values >= lo_95) & (actual_data.values <= hi_95)) * 100),
                'train_size': len(train_data),
                'status': 'OK'
            }
            windows.append(w)
        except Exception as e:
            windows.append({'origin_idx': origin, 'status': f'FAIL: {str(e)[:80]}'})

        origin += horizon

    return windows

@st.cache_data
def hw_forecast_future(series_values, _series_index, horizon,
                       alpha, beta, gamma, phi,
                       trend, seasonal, seasonal_periods, damped, optimized,
                       seed=42):
    """Generate future forecast beyond available data."""
    series_data = pd.Series(series_values, index=_series_index).asfreq('D')
    rng = np.random.default_rng(seed)

    try:
        mdl = ExponentialSmoothing(
            series_data,
            trend=trend, seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            damped_trend=damped
        )
        fit_kw = {}
        if optimized:
            fit_kw['optimized'] = True
        else:
            fit_kw['optimized'] = False
            fit_kw['smoothing_level'] = alpha
            fit_kw['smoothing_trend'] = beta
            fit_kw['smoothing_seasonal'] = gamma
            fit_kw['damping_trend'] = phi

        fit = mdl.fit(**fit_kw)
        fc = np.maximum(fit.forecast(horizon), 0)

        # Bootstrap CI
        resid = series_data.values - fit.fittedvalues.values
        valid_r = resid[~np.isnan(resid)]
        boot = np.zeros((1000, horizon))
        for b in range(1000):
            boot[b, :] = fc.values + rng.choice(valid_r, size=horizon, replace=True)
        lo_90 = np.percentile(boot, 5, axis=0)
        hi_90 = np.percentile(boot, 95, axis=0)
        lo_95 = np.percentile(boot, 2.5, axis=0)
        hi_95 = np.percentile(boot, 97.5, axis=0)

        fc_dates = pd.date_range(series_data.index[-1] + pd.Timedelta(days=1),
                                 periods=horizon, freq='D')

        return {
            'dates': fc_dates,
            'forecast': fc.values,
            'lo_90': lo_90, 'hi_90': hi_90,
            'lo_95': lo_95, 'hi_95': hi_95,
            'fitted': fit.fittedvalues,
            'aic': getattr(fit, 'aic', None),
            'bic': getattr(fit, 'bic', None),
            'status': 'OK'
        }
    except Exception as e:
        return {'status': f'FAIL: {str(e)}'}

# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.title("📈 HW Forecast")
st.sidebar.markdown("---")

# Data source
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.sidebar.info("👆 Select a CSV file to get started.")
    st.stop()

# Load data on button click — read bytes immediately to avoid stale file handles
if st.sidebar.button("📂 Load File", type="primary"):
    st.session_state['csv_bytes'] = uploaded_file.getvalue()
    st.session_state['csv_name'] = uploaded_file.name

if 'csv_bytes' not in st.session_state:
    st.sidebar.info("Click **Load File** after selecting your CSV.")
    st.stop()

try:
    series_full, nan_filled, raw_count = load_data(st.session_state['csv_bytes'])
    # Remove incomplete current month for backtesting
    current_date = series_full.index.max()
    last_complete = pd.Timestamp(current_date.year, current_date.month, 1) - pd.Timedelta(days=1)
    series = series_full[series_full.index <= last_complete].copy()
    # Store current (incomplete) month data for forecast vs actual comparison
    current_month_data = series_full[series_full.index > last_complete].copy()
    data_loaded = True
except Exception as e:
    st.sidebar.error(f"Error loading data: {e}")
    data_loaded = False
    st.stop()

st.sidebar.success(f"✅ {len(series):,} days loaded")
st.sidebar.caption(f"{series.index[0].date()} → {series.index[-1].date()}")
if len(current_month_data) > 0:
    st.sidebar.info(f"📅 Current month: {len(current_month_data)} days available ({current_month_data.index[0].date()} → {current_month_data.index[-1].date()})")

st.sidebar.markdown("---")
st.sidebar.subheader("Split Strategy")
split_option = st.sidebar.radio(
    "Train / Val / Test",
    ["70 / 15 / 15", "90 / 5 / 5", "Custom"],
    index=0
)

if split_option == "70 / 15 / 15":
    train_pct, val_pct, test_pct = 70, 15, 15
elif split_option == "90 / 5 / 5":
    train_pct, val_pct, test_pct = 90, 5, 5
else:
    train_pct = st.sidebar.slider("Train %", 50, 95, 80)
    val_pct = st.sidebar.slider("Val %", 2, 25, 10)
    test_pct = 100 - train_pct - val_pct
    if test_pct < 2:
        st.sidebar.error(f"Test% = {test_pct}% — too small. Reduce Train or Val.")
        st.stop()
    st.sidebar.caption(f"Test %: {test_pct}")

train, val, test_set, train_end_idx, val_end_idx = create_split(
    series, train_pct, val_pct, test_pct
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Parameters")

use_optimized = st.sidebar.checkbox("Auto-optimize (ETS)", value=False)

if not use_optimized:
    alpha = st.sidebar.slider("α (level)", 0.01, 0.99, 0.20, 0.01)
    beta = st.sidebar.slider("β (trend)", 0.01, 0.99, 0.20, 0.01)
    gamma = st.sidebar.slider("γ (seasonal)", 0.01, 0.50, 0.05, 0.01)
    phi = st.sidebar.slider("φ (damping)", 0.80, 0.99, 0.90, 0.01)
else:
    alpha, beta, gamma, phi = 0.2, 0.2, 0.05, 0.9  # defaults for display

trend_type = st.sidebar.selectbox("Trend", ["add", "mul", None], index=0)

# Auto-detect seasonal type
seasonal_period = st.sidebar.number_input("Seasonal Period (m)", 2, 365, 7)
auto_seasonal = st.sidebar.checkbox("Auto-detect Seasonal Type", value=True)
if auto_seasonal:
    det_type, det_corr, det_strength = detect_seasonal_type(train.values, period=seasonal_period)
    seasonal_type = det_type
    st.sidebar.caption(f"Detected: **{det_type}** (corr={det_corr:.3f}, strength={det_strength:.3f})")
else:
    seasonal_type = st.sidebar.selectbox("Seasonal", ["mul", "add", None], index=0)
use_damped = st.sidebar.checkbox("Damped Trend", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Forecast Settings")
horizon = st.sidebar.selectbox("Rolling Horizon (days)", [7, 14, 21, 28], index=0)
forecast_days = st.sidebar.number_input("Future Forecast Days", 7, 90, 28)

# ==============================================================================
# MAIN CONTENT — TABS
# ==============================================================================
st.title("🔮 Holt-Winters Rolling Forecast Dashboard")
st.caption("Pipeline : Data → Assumptions → Model → Backtest → Forecast")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data & Assumptions",
    "🔧 Model Tuning & Backtest",
    "🔬 Diagnostics & Stability",
    "📈 Results & Forecast",
    "📋 Split Comparison"
])

# ==============================================================================
# TAB 1: DATA & ASSUMPTIONS
# ==============================================================================
with tab1:
    st.header("Data Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Days", f"{len(series):,}")
    col2.metric("Mean", f"${series.mean():,.0f}")
    col3.metric("Std Dev", f"${series.std():,.0f}")
    col4.metric("CV", f"{series.std()/series.mean()*100:.1f}%")

    # Time series plot with splits
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=train.index, y=train.values,
        mode='lines', name='Train', line=dict(color='blue')))
    fig_ts.add_trace(go.Scatter(x=val.index, y=val.values,
        mode='lines', name='Validation', line=dict(color='orange')))
    fig_ts.add_trace(go.Scatter(x=test_set.index, y=test_set.values,
        mode='lines', name='Test', line=dict(color='red')))
    fig_ts.update_layout(
        title=f'Time Series with {train_pct}/{val_pct}/{test_pct} Split',
        xaxis_title='Date', yaxis_title='Approved USD ($)',
        height=400, hovermode='x unified'
    )
    st.plotly_chart(fig_ts, width='stretch')

    # Split details
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.subheader("Train")
        st.write(f"**{len(train):,} days** ({train.index[0].date()} → {train.index[-1].date()})")
        st.write(f"Mean: ${train.mean():,.0f}")
    with col_b:
        st.subheader("Validation")
        st.write(f"**{len(val):,} days** ({val.index[0].date()} → {val.index[-1].date()})")
        st.write(f"Mean: ${val.mean():,.0f}")
    with col_c:
        st.subheader("Test")
        st.write(f"**{len(test_set):,} days** ({test_set.index[0].date()} → {test_set.index[-1].date()})")
        st.write(f"Mean: ${test_set.mean():,.0f}")

    # Regime shift
    val_test_shift = (test_set.mean() - val.mean()) / val.mean() * 100
    if abs(val_test_shift) > 15:
        st.warning(f"⚠️ Val→Test shift: {val_test_shift:+.1f}% — significant regime change detected")
    else:
        st.success(f"✅ Val→Test shift: {val_test_shift:+.1f}% — moderate")

    st.markdown("---")
    st.header("Assumption Tests")

    # ADF Test
    adf = run_adf_test(series.values)
    col_adf1, col_adf2 = st.columns(2)
    with col_adf1:
        st.subheader("ADF Stationarity Test")
        st.write(f"**ADF Statistic:** {adf['statistic']:.4f}")
        st.write(f"**p-value:** {adf['p_value']:.6f}")
        st.write(f"**Lags:** {adf['lags']}")
        for k, v in adf['critical_values'].items():
            st.write(f"Critical {k}: {v:.4f}")
        if adf['p_value'] < 0.05:
            st.success("✅ Series is STATIONARY")
        else:
            st.info("ℹ️ Series is non-stationary — HW handles this via trend component")

    with col_adf2:
        st.subheader("Seasonal Decomposition")
        dec_model = st.selectbox("Decomposition Model", ["multiplicative", "additive"], index=0)

    trend_vals, seasonal_vals, resid_vals = run_decomposition(
        series.values, series.index, dec_model, seasonal_period
    )

    fig_dec = make_subplots(rows=4, cols=1,
        subplot_titles=['Observed', 'Trend', 'Seasonal', 'Residual'],
        shared_xaxes=True, vertical_spacing=0.05)
    fig_dec.add_trace(go.Scatter(x=series.index, y=series.values, name='Observed',
        line=dict(color='gray')), row=1, col=1)
    fig_dec.add_trace(go.Scatter(x=series.index, y=trend_vals, name='Trend',
        line=dict(color='blue')), row=2, col=1)
    fig_dec.add_trace(go.Scatter(x=series.index, y=seasonal_vals, name='Seasonal',
        line=dict(color='green')), row=3, col=1)
    fig_dec.add_trace(go.Scatter(x=series.index, y=resid_vals, name='Residual',
        line=dict(color='red')), row=4, col=1)
    fig_dec.update_layout(height=600, showlegend=False,
        title=f'Seasonal Decomposition ({dec_model}, m={seasonal_period})')
    st.plotly_chart(fig_dec, width='stretch')

# ==============================================================================
# TAB 2: MODEL TUNING & BACKTEST
# ==============================================================================
with tab2:
    st.header("Model Configuration")

    # Show current params
    params_col1, params_col2 = st.columns(2)
    with params_col1:
        st.subheader("Current Parameters")
        if use_optimized:
            st.info("🤖 Auto-optimized by statsmodels (ETS)")
        else:
            st.write(f"**α (level):** {alpha}")
            st.write(f"**β (trend):** {beta}")
            st.write(f"**γ (seasonal):** {gamma}")
            st.write(f"**φ (damping):** {phi}")
        st.write(f"**Trend:** {trend_type}")
        st.write(f"**Seasonal:** {seasonal_type}")
        st.write(f"**Period (m):** {seasonal_period}")
        st.write(f"**Damped:** {use_damped}")

    with params_col2:
        st.subheader("Backtest Config")
        st.write(f"**Split:** {train_pct}/{val_pct}/{test_pct}")
        st.write(f"**Horizon:** {horizon} days")
        st.write(f"**Test period:** {test_set.index[0].date()} → {test_set.index[-1].date()}")
        st.write(f"**Test days:** {len(test_set)}")

    st.markdown("---")

    # Run backtest
    if st.button("🚀 Run Rolling Backtest", type="primary"):
        _prog_bt = st.empty()
        _prog_bt.progress(0, text="Initializing backtest...")
        t0 = time.time()
        _prog_bt.progress(10, text="Fitting models & bootstrapping CIs...")
        windows = hw_rolling_backtest(
            series.values, series.index,
            val_end_idx, horizon,
            alpha, beta, gamma, phi,
            trend_type, seasonal_type, seasonal_period, use_damped, use_optimized
        )
        _prog_bt.progress(90, text="Computing metrics...")
        elapsed = time.time() - t0

        ok_windows = [w for w in windows if w['status'] == 'OK']
        fail_windows = [w for w in windows if w['status'] != 'OK']

        st.session_state['backtest_windows'] = windows
        st.session_state['backtest_ok'] = ok_windows
        st.session_state['backtest_elapsed'] = elapsed
        st.session_state['backtest_params'] = {
            'alpha': alpha, 'beta': beta, 'gamma': gamma, 'phi': phi,
            'optimized': use_optimized, 'horizon': horizon,
            'split': f"{train_pct}/{val_pct}/{test_pct}"
        }

        _prog_bt.progress(100, text="Backtest complete!")
        time.sleep(0.5)
        _prog_bt.empty()
        st.success(f"✅ Backtest complete: {len(ok_windows)}/{len(windows)} windows OK in {elapsed:.1f}s")

        if fail_windows:
            st.warning(f"⚠️ {len(fail_windows)} windows failed")

    # Show results if available
    if 'backtest_ok' in st.session_state and st.session_state['backtest_ok']:
        ok = st.session_state['backtest_ok']

        # Aggregate metrics
        all_actual = np.concatenate([np.array(w['actual']) for w in ok])
        all_fc = np.concatenate([np.array(w['forecast']) for w in ok])
        agg = compute_all_metrics(all_actual, all_fc, train.values)

        st.subheader("Aggregate Metrics")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("sMAPE", f"{agg['sMAPE']:.2f}%",
                  delta="Good" if agg['sMAPE'] < 15 else "High" if agg['sMAPE'] < 25 else "⚠️ Very High")
        m2.metric("MASE", f"{agg['MASE']:.3f}",
                  delta="Beats naive" if agg['MASE'] < 1.0 else "Below naive")
        m3.metric("R²", f"{agg['R²']:.4f}")
        m4.metric("RMSE", f"${agg['RMSE']:,.0f}")
        m5.metric("Bias", f"${agg['Bias']:+,.0f}")

        # Per-window table
        st.subheader("Per-Window Results")
        rows = []
        for i, w in enumerate(ok, 1):
            rows.append({
                'Window': i,
                'Period': f"{w['forecast_start'][:10]} → {w['forecast_end'][:10]}",
                'sMAPE': f"{w['smape']:.2f}%",
                'MASE': f"{w['mase']:.3f}",
                'RMSE': f"${w['rmse']:,.0f}",
                'Bias': f"${w['bias']:+,.0f}",
                'CI95%': f"{w['ci95_coverage']:.0f}%",
                'Train Size': w['train_size']
            })
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

        # Backtest plot
        st.subheader("Backtest Visualization")
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=test_set.index, y=test_set.values,
            mode='lines', name='Actual', line=dict(color='black', width=2)
        ))

        colors = ['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] * 5
        for i, w in enumerate(ok):
            dates = pd.to_datetime(w['dates'])
            showlegend = (i == 0)
            fig_bt.add_trace(go.Scatter(
                x=dates, y=w['forecast'],
                mode='lines+markers',
                name='Rolling FC' if showlegend else None,
                showlegend=showlegend,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=4)
            ))
            fig_bt.add_trace(go.Scatter(
                x=list(dates) + list(dates)[::-1],
                y=w['hi_95'] + w['lo_95'][::-1],
                fill='toself', fillcolor=f'rgba(0,200,0,0.08)',
                line=dict(color='rgba(0,0,0,0)'), showlegend=False
            ))

        fig_bt.update_layout(
            title=f'Rolling Backtest h={horizon} (sMAPE={agg["sMAPE"]:.1f}%, MASE={agg["MASE"]:.3f})',
            xaxis_title='Date', yaxis_title='USD ($)',
            height=450, hovermode='x unified'
        )
        st.plotly_chart(fig_bt, width='stretch')

        # Accuracy decay
        st.subheader("Accuracy Decay (Day-by-Day)")
        decay_data = []
        for d in range(horizon):
            day_actual = [w['actual'][d] for w in ok if len(w['actual']) > d]
            day_fc = [w['forecast'][d] for w in ok if len(w['forecast']) > d]
            if day_actual:
                day_sm = smape(np.array(day_actual), np.array(day_fc))
                day_mae = np.mean(np.abs(np.array(day_actual) - np.array(day_fc)))
                decay_data.append({'Day': f'd+{d+1}', 'sMAPE': day_sm, 'MAE': day_mae})

        if decay_data:
            df_decay = pd.DataFrame(decay_data)
            fig_decay = go.Figure()
            fig_decay.add_trace(go.Bar(
                x=df_decay['Day'], y=df_decay['sMAPE'],
                marker_color=['green' if s < 15 else 'orange' if s < 25 else 'red'
                              for s in df_decay['sMAPE']],
                text=[f"{s:.1f}%" for s in df_decay['sMAPE']],
                textposition='outside'
            ))
            max_val = df_decay['sMAPE'].max()
            fig_decay.update_layout(title='sMAPE by Forecast Day',
                yaxis_title='sMAPE %', height=400,
                yaxis=dict(range=[0, max_val * 1.25]),
                uniformtext_minsize=10, uniformtext_mode='show')
            st.plotly_chart(fig_decay, width='stretch')

# ==============================================================================
# TAB 3: DIAGNOSTICS & STABILITY
# ==============================================================================
with tab3:
    st.header("Residual Diagnostics & CV Stability")
    st.caption("Run a backtest first (Tab 2), then analyze residuals and model stability here.")

    if 'backtest_ok' in st.session_state and st.session_state['backtest_ok']:
        ok = st.session_state['backtest_ok']
        all_actual = np.concatenate([np.array(w['actual']) for w in ok])
        all_fc = np.concatenate([np.array(w['forecast']) for w in ok])

        # ---- RESIDUAL DIAGNOSTICS ----
        st.subheader("🔬 Residual Diagnostics")
        diag = run_residual_diagnostics(all_actual, all_fc)
        st.session_state['diagnostics'] = diag

        d1, d2 = st.columns(2)
        with d1:
            icon_sw = "✅" if diag['sw_pass'] else "❌"
            st.metric("Shapiro-Wilk (Normality)", f"{icon_sw} {'PASS' if diag['sw_pass'] else 'FAIL'}")
            st.caption(f"Statistic: {diag['sw_stat']:.4f}, p={diag['sw_p']:.4f}")
            if diag['sw_pass']:
                st.success("Residuals are normally distributed")
            else:
                st.info("Non-normal residuals — CIs may be approximate (bootstrap helps)")
        with d2:
            icon_bi = "✅" if diag['bias_pass'] else "❌"
            st.metric("Systematic Bias", f"{icon_bi} {diag['bias_pct']:.1f}%")
            st.caption(f"Mean residual: ${diag['bias']:+,.0f}")
            if diag['bias_pass']:
                st.success("Bias < 5% — acceptable")
            else:
                st.warning(f"Bias is {diag['bias_pct']:.1f}% — model systematically {'over' if diag['bias'] < 0 else 'under'}-predicts")

        # Additional stats
        stat_c1, stat_c2, stat_c3 = st.columns(3)
        stat_c1.metric("Residual Std Dev", f"${diag['std']:,.0f}")
        stat_c2.metric("Skewness", f"{diag['skew']:.3f}",
                       delta="Symmetric" if abs(diag['skew']) < 0.5 else "Skewed")
        stat_c3.metric("Excess Kurtosis", f"{diag['kurtosis']:.3f}",
                       delta="Normal tails" if abs(diag['kurtosis']) < 1 else "Heavy tails")

        # ---- LJUNG-BOX SUMMARY (Per-Window, computed here) ----
        st.markdown("---")
        st.subheader("📊 Ljung-Box Summary")

        # Run per-window diagnostics directly
        windows_for_cache = [{
            'actual': w['actual'], 'forecast': w['forecast'],
            'forecast_start': w['forecast_start'], 'forecast_end': w['forecast_end'],
            'status': w['status']
        } for w in ok]
        pw_results = run_per_window_diagnostics(windows_for_cache)
        st.session_state['per_window_diag'] = pw_results

        if pw_results:
            n_windows = len(pw_results)
            lb_pass_count = sum(1 for r in pw_results if r['LB Pass'])
            acf_pass_count = sum(1 for r in pw_results if r['ACF Pass'])
            lb_pass_pct = lb_pass_count / n_windows * 100
            acf_pass_pct = acf_pass_count / n_windows * 100
            overall_pw_pass = lb_pass_pct >= 60

            st.session_state['pw_lb_pass'] = overall_pw_pass
            st.session_state['pw_lb_pct'] = lb_pass_pct

            lb_s1, lb_s2, lb_s3 = st.columns(3)
            lb_s1.metric("Windows Tested", n_windows)
            lb_s2.metric("LB Pass Rate", f"{lb_pass_pct:.0f}%",
                         delta="PASS" if overall_pw_pass else "FAIL")
            lb_s3.metric("ACF Pass Rate", f"{acf_pass_pct:.0f}%",
                         delta="PASS" if acf_pass_pct >= 60 else "FAIL")

            if overall_pw_pass:
                st.success(
                    f"✅ Per-window Ljung-Box: **{lb_pass_count}/{n_windows}** windows pass ({lb_pass_pct:.0f}%) — "
                    f"no systematic autocorrelation within individual forecast windows."
                )
            else:
                st.warning(
                    f"⚠️ Per-window Ljung-Box: **{lb_pass_count}/{n_windows}** windows pass ({lb_pass_pct:.0f}%) — "
                    f"some windows show within-window autocorrelation."
                )

            # Expandable detail: table + chart
            with st.expander("🔍 Per-Window Detail (Table & Chart)", expanded=False):
                st.dataframe(pd.DataFrame(pw_results), width='stretch', hide_index=True)

                lbp_vals = [r['LB min-p'] for r in pw_results]
                fig_lbp = go.Figure()
                fig_lbp.add_trace(go.Bar(
                    x=[f"W{r['Window']}" for r in pw_results],
                    y=lbp_vals,
                    marker_color=['green' if v > 0.05 else 'red' for v in lbp_vals],
                    text=[f"{v:.3f}" for v in lbp_vals],
                    textposition='outside'
                ))
                fig_lbp.add_hline(y=0.05, line_dash='dash', line_color='red',
                    annotation_text='α=0.05')
                max_lbp = max(lbp_vals) if lbp_vals else 1
                fig_lbp.update_layout(
                    title='Ljung-Box p-value per Window (green = pass, red = fail)',
                    height=350, yaxis_title='p-value',
                    yaxis=dict(range=[0, min(max_lbp * 1.3, 1.05)])
                )
                st.plotly_chart(fig_lbp, width='stretch')

        # ACF Plot + Residual charts (expandable)
        with st.expander("📉 Residual ACF & Distribution", expanded=False):
            st.subheader("Residual ACF")
            acf_v = diag['acf_vals']
            n_obs = len(diag['residuals'])
            ci_bound = 1.96 / np.sqrt(n_obs)
            fig_acf = go.Figure()
            fig_acf.add_trace(go.Bar(
                x=list(range(len(acf_v))), y=acf_v,
                marker_color=['red' if abs(v) > ci_bound and i > 0 else '#1f77b4'
                              for i, v in enumerate(acf_v)],
                name='ACF'
            ))
            fig_acf.add_hline(y=ci_bound, line_dash='dash', line_color='gray')
            fig_acf.add_hline(y=-ci_bound, line_dash='dash', line_color='gray')
            fig_acf.add_hline(y=0, line_color='black', line_width=0.5)
            fig_acf.update_layout(title='Autocorrelation of Residuals',
                xaxis_title='Lag', yaxis_title='ACF', height=300,
                yaxis=dict(range=[-0.5, 1.0]))
            st.plotly_chart(fig_acf, width='stretch', key='acf_tab3')

            # Residual distribution
            col_hist, col_time = st.columns(2)
            resid_arr = np.array(diag['residuals'])
            with col_hist:
                st.subheader("Residual Distribution")
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=resid_arr, nbinsx=40,
                    marker_color='#1f77b4', opacity=0.7, name='Residuals'))
                fig_hist.add_vline(x=0, line_dash='dash', line_color='red')
                fig_hist.add_vline(x=np.mean(resid_arr), line_dash='dot', line_color='orange',
                    annotation_text=f'Mean=${np.mean(resid_arr):+,.0f}')
                fig_hist.update_layout(height=300, xaxis_title='Residual ($)', yaxis_title='Count')
                st.plotly_chart(fig_hist, width='stretch')
            with col_time:
                st.subheader("Residuals Over Forecast Horizon")
                fig_rtime = go.Figure()
                fig_rtime.add_trace(go.Scatter(y=resid_arr, mode='lines', line=dict(color='#1f77b4', width=1),
                    name='Residual'))
                fig_rtime.add_hline(y=0, line_dash='dash', line_color='red')
                fig_rtime.update_layout(height=300, xaxis_title='Observation', yaxis_title='Residual ($)')
                st.plotly_chart(fig_rtime, width='stretch')

        # ---- CV STABILITY CHECK ----
        st.markdown("---")
        st.subheader("🔄 Cross-Validation Stability")
        st.caption("Expanding-window CV on TRAIN data — tests if model performance is consistent across time periods.")

        n_cv_folds = st.slider("Number of CV Folds", 3, 10, 7)

        if st.button("🔄 Run CV Stability Check", type="primary"):
            with st.spinner(f"Running {n_cv_folds}-fold expanding CV..."):
                cv_results = run_cv_stability(
                    train.values, train.index, n_cv_folds,
                    alpha, beta, gamma, phi,
                    trend_type, seasonal_type, seasonal_period, use_damped, use_optimized,
                    horizon=horizon
                )
            st.session_state['cv_stability'] = cv_results
            st.success(f"✅ CV complete: {sum(1 for r in cv_results if r['status'] == 'OK')}/{len(cv_results)} folds OK")

        if 'cv_stability' in st.session_state:
            cv_res = st.session_state['cv_stability']
            ok_folds = [r for r in cv_res if r['status'] == 'OK']

            if ok_folds:
                cv_smapes = [r['smape'] for r in ok_folds]
                cv_mean = np.mean(cv_smapes)
                cv_std = np.std(cv_smapes)
                cv_pct = (cv_std / cv_mean * 100) if cv_mean > 0 else 999

                # Stability classification
                if cv_pct < 15:
                    stab_label, stab_color = '✅ STABLE', 'success'
                elif cv_pct < 30:
                    stab_label, stab_color = '⚠️ MODERATE', 'warning'
                elif cv_pct < 50:
                    stab_label, stab_color = '⚠️ UNSTABLE', 'warning'
                else:
                    stab_label, stab_color = '❌ HIGHLY UNSTABLE', 'error'

                cv_c1, cv_c2, cv_c3, cv_c4 = st.columns(4)
                cv_c1.metric("CV Mean sMAPE", f"{cv_mean:.2f}%")
                cv_c2.metric("CV Std Dev", f"{cv_std:.2f}%")
                cv_c3.metric("CV %", f"{cv_pct:.1f}%")
                cv_c4.metric("Stability", stab_label.split(' ', 1)[1])

                getattr(st, stab_color)(f"{stab_label} — CV coefficient of variation = {cv_pct:.1f}%")

                # Per-fold table
                st.dataframe(pd.DataFrame(ok_folds), width='stretch', hide_index=True)

                # Per-fold bar chart
                fig_cv = go.Figure()
                fig_cv.add_trace(go.Bar(
                    x=[f"Fold {r['fold']}" for r in ok_folds],
                    y=[r['smape'] for r in ok_folds],
                    marker_color=['green' if r['smape'] < 15 else 'orange' if r['smape'] < 25 else 'red'
                                  for r in ok_folds],
                    text=[f"{r['smape']:.1f}%" for r in ok_folds],
                    textposition='outside'
                ))
                fig_cv.add_hline(y=cv_mean, line_dash='dash', line_color='blue',
                    annotation_text=f'Mean={cv_mean:.1f}%')
                max_cv = max(r['smape'] for r in ok_folds)
                fig_cv.update_layout(title=f'CV Stability — sMAPE per Fold (CV={cv_pct:.1f}%)',
                    yaxis_title='sMAPE %', height=350,
                    yaxis=dict(range=[0, max_cv * 1.3]))
                st.plotly_chart(fig_cv, width='stretch')

                # Store for scorecard
                st.session_state['cv_pct'] = cv_pct
                st.session_state['cv_stab_label'] = stab_label
            else:
                st.error("All CV folds failed.")
    else:
        st.info("👈 Run a backtest in Tab 2 first, then come here for diagnostics.")

# ==============================================================================
# TAB 4: RESULTS & FORECAST
# ==============================================================================
with tab4:
    st.header("Production Forecast")

    st.info(f"Forecast **{forecast_days} days** ahead from {series.index[-1].date()}")

    if st.button("🔮 Generate Forecast", type="primary"):
        _prog_fc = st.empty()
        _prog_fc.progress(0, text="Preparing forecast model...")
        _prog_fc.progress(15, text="Fitting model & bootstrapping 1000 paths...")
        result = hw_forecast_future(
            series.values, series.index, forecast_days,
            alpha, beta, gamma, phi,
            trend_type, seasonal_type, seasonal_period, use_damped, use_optimized
        )
        _prog_fc.progress(95, text="Finalizing...")

        if result['status'] == 'OK':
            st.session_state['forecast'] = result
            _prog_fc.progress(100, text="Forecast complete!")
            time.sleep(0.5)
            _prog_fc.empty()
            st.success("✅ Forecast generated successfully!")
        else:
            _prog_fc.empty()
            st.error(f"Forecast failed: {result['status']}")

    if 'forecast' in st.session_state:
        fc = st.session_state['forecast']

        # Find overlap between forecast dates and current month actual data
        fc_dates_index = pd.DatetimeIndex(fc['dates'])
        overlap_mask = fc_dates_index.isin(current_month_data.index)
        has_overlap = overlap_mask.any()

        if has_overlap:
            overlap_dates = fc_dates_index[overlap_mask]
            overlap_actual = current_month_data.loc[overlap_dates].values
            overlap_forecast = fc['forecast'][overlap_mask]
            overlap_lo95 = fc['lo_95'][overlap_mask]
            overlap_hi95 = fc['hi_95'][overlap_mask]

        # Forecast chart
        history_tail = series.iloc[-60:]
        fig_fc = go.Figure()

        fig_fc.add_trace(go.Scatter(
            x=history_tail.index, y=history_tail.values,
            mode='lines', name='Historical', line=dict(color='gray', width=2)
        ))
        fig_fc.add_trace(go.Scatter(
            x=fc['dates'], y=fc['forecast'],
            mode='lines+markers', name='Forecast',
            line=dict(color='green', width=2.5), marker=dict(size=5)
        ))
        fig_fc.add_trace(go.Scatter(
            x=list(fc['dates']) + list(fc['dates'])[::-1],
            y=list(fc['hi_95']) + list(fc['lo_95'])[::-1],
            fill='toself', fillcolor='rgba(0,200,0,0.1)',
            line=dict(color='rgba(0,0,0,0)'), name='95% CI'
        ))
        fig_fc.add_trace(go.Scatter(
            x=list(fc['dates']) + list(fc['dates'])[::-1],
            y=list(fc['hi_90']) + list(fc['lo_90'])[::-1],
            fill='toself', fillcolor='rgba(0,200,0,0.15)',
            line=dict(color='rgba(0,0,0,0)'), name='90% CI'
        ))

        # Overlay actual current month data
        if has_overlap:
            fig_fc.add_trace(go.Scatter(
                x=overlap_dates, y=overlap_actual,
                mode='lines+markers', name='Actual (Current Month)',
                line=dict(color='#FF6B6B', width=2.5, dash='solid'),
                marker=dict(size=7, symbol='diamond')
            ))

        fig_fc.add_vline(x=series.index[-1], line_dash='dash', line_color='red')
        fig_fc.update_layout(
            title=f'Forecast: {forecast_days} Days Ahead' + (' — with Actual Overlay' if has_overlap else ''),
            xaxis_title='Date', yaxis_title='USD ($)',
            height=450, hovermode='x unified'
        )
        st.plotly_chart(fig_fc, width='stretch')

        # ---- FORECAST vs ACTUAL COMPARISON ----
        if has_overlap:
            st.markdown("---")
            st.subheader("📊 Forecast vs Actual (Current Month)")
            st.caption(f"Comparing {len(overlap_dates)} days of actual data against the forecast.")

            # Metrics
            overlap_errors = overlap_actual - overlap_forecast
            overlap_smape = smape(overlap_actual, overlap_forecast)
            overlap_mae = mean_absolute_error(overlap_actual, overlap_forecast)
            overlap_mape = np.mean(np.abs(overlap_errors) / np.abs(overlap_actual)) * 100
            overlap_bias = np.mean(overlap_errors)
            overlap_bias_pct = abs(overlap_bias) / np.mean(overlap_actual) * 100
            within_ci = np.sum((overlap_actual >= overlap_lo95) & (overlap_actual <= overlap_hi95))
            ci_coverage = within_ci / len(overlap_actual) * 100

            ov_c1, ov_c2, ov_c3, ov_c4 = st.columns(4)
            ov_c1.metric("sMAPE", f"{overlap_smape:.2f}%",
                         delta="Good" if overlap_smape < 15 else "Fair" if overlap_smape < 25 else "High")
            ov_c2.metric("MAE", f"${overlap_mae:,.0f}")
            ov_c3.metric("Bias", f"{overlap_bias_pct:.1f}%",
                         delta=f"{'Over' if overlap_bias < 0 else 'Under'}-predicted")
            ov_c4.metric("CI95 Coverage", f"{ci_coverage:.0f}%",
                         delta="PASS" if ci_coverage >= 70 else "FAIL")

            if overlap_smape < 15:
                st.success(f"✅ Forecast is tracking actuals well — sMAPE = {overlap_smape:.2f}%")
            elif overlap_smape < 25:
                st.info(f"ℹ️ Forecast is reasonably close — sMAPE = {overlap_smape:.2f}%")
            else:
                st.warning(f"⚠️ Forecast diverging from actuals — sMAPE = {overlap_smape:.2f}%")

            # Daily comparison table
            df_comparison = pd.DataFrame({
                'Date': overlap_dates.strftime('%Y-%m-%d'),
                'Day': overlap_dates.strftime('%a'),
                'Actual': [f"${v:,.0f}" for v in overlap_actual],
                'Forecast': [f"${v:,.0f}" for v in overlap_forecast],
                'Error': [f"${v:+,.0f}" for v in overlap_errors],
                'Error %': [f"{abs(e)/a*100:.1f}%" if a != 0 else "N/A"
                            for e, a in zip(overlap_errors, overlap_actual)],
                'In CI95': ['✅' if lo <= a <= hi else '❌'
                            for a, lo, hi in zip(overlap_actual, overlap_lo95, overlap_hi95)]
            })
            st.dataframe(df_comparison, width='stretch', hide_index=True)

            # Cumulative comparison
            cum_c1, cum_c2, cum_c3 = st.columns(3)
            cum_c1.metric("Total Actual", f"${np.sum(overlap_actual):,.0f}")
            cum_c2.metric("Total Forecast", f"${np.sum(overlap_forecast):,.0f}")
            cum_diff = np.sum(overlap_actual) - np.sum(overlap_forecast)
            cum_c3.metric("Cumulative Difference", f"${cum_diff:+,.0f}",
                          delta=f"{abs(cum_diff)/np.sum(overlap_actual)*100:.1f}%")
        else:
            st.info("📅 No current month data available to compare against the forecast.")

        # Forecast table
        col_fc1, col_fc2 = st.columns(2)
        with col_fc1:
            st.subheader("Forecast Summary")
            st.metric("Total Forecast", f"${np.sum(fc['forecast']):,.0f}")
            st.metric("Daily Average", f"${np.mean(fc['forecast']):,.0f}")
            st.metric("Min Day", f"${np.min(fc['forecast']):,.0f}")
            st.metric("Max Day", f"${np.max(fc['forecast']):,.0f}")

        with col_fc2:
            st.subheader("Daily Forecast")
            df_fc_table = pd.DataFrame({
                'Date': fc['dates'].strftime('%Y-%m-%d'),
                'Day': fc['dates'].strftime('%a'),
                'Forecast': [f"${v:,.0f}" for v in fc['forecast']],
                'Lo 95%': [f"${v:,.0f}" for v in fc['lo_95']],
                'Hi 95%': [f"${v:,.0f}" for v in fc['hi_95']]
            })
            st.dataframe(df_fc_table, width='stretch', hide_index=True)

        # Download
        df_download = pd.DataFrame({
            'date': fc['dates'],
            'forecast': fc['forecast'],
            'lo_90': fc['lo_90'], 'hi_90': fc['hi_90'],
            'lo_95': fc['lo_95'], 'hi_95': fc['hi_95']
        })
        csv = df_download.to_csv(index=False)
        st.download_button("📥 Download Forecast CSV", csv,
                           "forecast.csv", "text/csv")

    # Production readiness (if backtest ran)
    if 'backtest_ok' in st.session_state and st.session_state['backtest_ok']:
        st.markdown("---")
        st.header("🚦 Forecast Deployment Scorecard")
        st.caption(" — Run Diagnostics & CV (Tab 3) for full scoring.")

        ok = st.session_state['backtest_ok']
        all_actual = np.concatenate([np.array(w['actual']) for w in ok])
        all_fc = np.concatenate([np.array(w['forecast']) for w in ok])
        agg = compute_all_metrics(all_actual, all_fc, train.values)
        avg_ci95 = np.mean([w['ci95_coverage'] for w in ok])
        sm_cv = np.std([w['smape'] for w in ok]) / np.mean([w['smape'] for w in ok]) * 100

        # Core criteria
        criteria = [
            ("📊 sMAPE < 20%", agg['sMAPE'] < 20, f"{agg['sMAPE']:.2f}%", "Primary accuracy metric"),
            ("📊 MASE < 1.5", agg['MASE'] < 1.5, f"{agg['MASE']:.3f}", "Must beat seasonal naive"),
            ("📊 R² > 0", agg['R²'] > 0, f"{agg['R²']:.4f}", "Better than mean predictor"),
            ("📊 Window CV < 50%", sm_cv < 50, f"{sm_cv:.1f}%", "Consistent across windows"),
            ("📊 CI95 Coverage > 70%", avg_ci95 > 70, f"{avg_ci95:.1f}%", "Confidence intervals reliable"),
        ]

        # Add diagnostics criteria if available
        if 'diagnostics' in st.session_state:
            diag = st.session_state['diagnostics']
            # Prefer per-window LB over concatenated LB when available
            if 'pw_lb_pass' in st.session_state:
                pw_pass = st.session_state['pw_lb_pass']
                pw_pct = st.session_state['pw_lb_pct']
                criteria.append(("🔍 Per-Window LB (autocorr)", pw_pass,
                    f"{pw_pct:.0f}% pass", "Per-window test — fair for rolling backtests"))
            else:
                criteria.append(("🔬 Ljung-Box (no autocorr)", diag['lb_pass'],
                    f"p={diag['lb_pvals'][0]:.4f}", "Concatenated test — run Tab 4 for fairer test"))
            criteria.append(("🔬 Bias < 5%", diag['bias_pass'],
                f"{diag['bias_pct']:.1f}%", "No systematic over/under-prediction"))

        # Add CV stability if available
        if 'cv_pct' in st.session_state:
            cv_p = st.session_state['cv_pct']
            criteria.append(("🔄 CV Stability < 30%", cv_p < 30,
                f"{cv_p:.1f}%", "Model is stable across time periods"))

        n_total = len(criteria)
        n_pass = sum(1 for _, p, _, _ in criteria if p)

        # Scorecard table
        score_rows = []
        for name, passed, val, note in criteria:
            score_rows.append({
                'Criterion': name,
                'Status': '✅ PASS' if passed else '❌ FAIL',
                'Value': val,
                'Note': note
            })
        st.dataframe(pd.DataFrame(score_rows), width='stretch', hide_index=True)

        # Overall verdict
        pct_pass = n_pass / n_total * 100
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Score", f"{n_pass}/{n_total}")
        sc2.metric("Pass Rate", f"{pct_pass:.0f}%")

        if pct_pass >= 80:
            sc3.metric("Status", "🟢 PRODUCTION")
            st.success(f"🟢 PRODUCTION READY — {n_pass}/{n_total} criteria passed ({pct_pass:.0f}%). Deploy with monitoring.")
        elif pct_pass >= 60:
            sc3.metric("Status", "🟡 CONDITIONAL")
            st.warning(f"🟡 CONDITIONAL APPROVAL — {n_pass}/{n_total} passed. Deploy with human oversight.")
        elif pct_pass >= 40:
            sc3.metric("Status", "🟠 LIMITED")
            st.warning(f"🟠 LIMITED USE — {n_pass}/{n_total} passed. Use for directional guidance only.")
        else:
            sc3.metric("Status", "🔴 NOT READY")
            st.error(f"🔴 NOT PRODUCTION READY — {n_pass}/{n_total} passed. Investigate parameters, split, or data issues.")

        # Recommendations
        fails = [(name, val, note) for name, passed, val, note in criteria if not passed]
        if fails:
            st.subheader("Recommendations")
            for name, val, note in fails:
                st.write(f"• **{name}** = {val} — {note}")

# ==============================================================================
# TAB 5: SPLIT COMPARISON
# ==============================================================================
with tab5:
    st.header("Split Strategy Comparison")
    st.caption("Compare 70/15/15 vs 90/5/5 side-by-side using the same model parameters")

    if st.button("⚡ Run Both Splits", type="primary"):
        results_both = {}
        _prog_sp = st.empty()
        _prog_sp.progress(0, text="Starting split comparison...")
        step = 0
        total_steps = 4  # 2 splits × 2 horizons

        for label, (tp, vp, tep) in [("70/15/15", (70, 15, 15)), ("90/5/5", (90, 5, 5))]:
            tr, va, te, tr_end, va_end = create_split(series, tp, vp, tep)

            for h in [7, 14]:
                step += 1
                _prog_sp.progress(
                    int(step / total_steps * 90),
                    text=f"Running {label} h={h}... ({step}/{total_steps})"
                )
                windows = hw_rolling_backtest(
                    series.values, series.index,
                    va_end, h,
                    alpha, beta, gamma, phi,
                    trend_type, seasonal_type, seasonal_period, use_damped, use_optimized,
                    seed=42
                )
                ok_w = [w for w in windows if w['status'] == 'OK']
                if ok_w:
                    all_a = np.concatenate([np.array(w['actual']) for w in ok_w])
                    all_f = np.concatenate([np.array(w['forecast']) for w in ok_w])
                    m = compute_all_metrics(all_a, all_f, tr.values)
                    avg_ci = np.mean([w['ci95_coverage'] for w in ok_w])
                    results_both[f"{label}_h{h}"] = {
                        'split': label, 'horizon': h,
                        'smape': m['sMAPE'], 'mase': m['MASE'],
                        'mape': m['MAPE'], 'r2': m['R²'],
                        'rmse': m['RMSE'], 'bias': m['Bias'],
                        'ci95': avg_ci, 'n_days': len(all_a),
                        'n_windows': len(ok_w),
                        'test_start': str(te.index[0].date()),
                        'test_end': str(te.index[-1].date())
                    }

        st.session_state['split_comparison'] = results_both
        _prog_sp.progress(100, text="Split comparison complete!")
        time.sleep(0.5)
        _prog_sp.empty()
        st.success("✅ Both splits completed!")

    if 'split_comparison' in st.session_state:
        results = st.session_state['split_comparison']

        # Comparison table
        df_comp = pd.DataFrame(results.values())
        st.subheader("Metrics Comparison")
        st.dataframe(
            df_comp[['split', 'horizon', 'smape', 'mase', 'mape', 'r2', 'rmse', 'bias', 'ci95', 'n_days']].style.format({
                'smape': '{:.2f}%', 'mase': '{:.3f}', 'mape': '{:.2f}%',
                'r2': '{:.4f}', 'rmse': '${:,.0f}', 'bias': '${:+,.0f}',
                'ci95': '{:.0f}%'
            }),
            width='stretch', hide_index=True
        )

        # Bar chart comparison
        fig_comp = make_subplots(rows=1, cols=3,
            subplot_titles=['sMAPE (lower=better)', 'MASE (< 1 beats naive)', 'R² (higher=better)'])

        for metric_idx, metric in enumerate(['smape', 'mase', 'r2'], 1):
            for split_label, color in [('70/15/15', '#1f77b4'), ('90/5/5', '#ff7f0e')]:
                sub = df_comp[df_comp['split'] == split_label]
                fig_comp.add_trace(go.Bar(
                    x=[f"h={r['horizon']}" for _, r in sub.iterrows()],
                    y=sub[metric],
                    name=split_label if metric_idx == 1 else None,
                    showlegend=(metric_idx == 1),
                    marker_color=color,
                    text=[f"{v:.2f}" for v in sub[metric]],
                    textposition='outside'
                ), row=1, col=metric_idx)

            if metric == 'mase':
                fig_comp.add_hline(y=1.0, line_dash='dash', line_color='red', row=1, col=metric_idx)

        fig_comp.update_layout(title='Split Strategy Comparison', height=400, barmode='group')
        st.plotly_chart(fig_comp, width='stretch')

        # Winner
        st.subheader("Verdict")
        best_row = df_comp.loc[df_comp['smape'].idxmin()]
        st.success(
            f"🏆 **Winner:** {best_row['split']} with h={int(best_row['horizon'])} "
            f"— sMAPE={best_row['smape']:.2f}%, MASE={best_row['mase']:.3f}"
        )

        best_70 = df_comp[df_comp['split'] == '70/15/15']['smape'].min()
        best_90 = df_comp[df_comp['split'] == '90/5/5']['smape'].min()
        diff = best_70 - best_90

        if diff > 0:
            st.info(f"90/5/5 wins by {diff:.2f}pp — more training data helps")
        elif diff < 0:
            st.info(f"70/15/15 wins by {-diff:.2f}pp — larger test set gives more reliable evaluation")
        else:
            st.info("Both strategies are tied")

# ==============================================================================
# FOOTER
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.caption(f"Last run: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
st.sidebar.caption("Holt-Winters / by JC - Jose Sirias")
