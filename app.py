import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from io import BytesIO

st.set_page_config(page_title="Holt-Winters Forecast", page_icon="📈", layout="wide")

st.title("📈 Holt-Winters Forecast")
st.markdown("Upload a CSV file with time-series data and generate forecasts using the Holt-Winters Exponential Smoothing method.")

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    sep = st.sidebar.selectbox("CSV separator", [",", ";", "\t", "|"], index=0)
    try:
        df = pd.read_csv(uploaded_file, sep=sep)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    columns = df.columns.tolist()

    date_col = st.sidebar.selectbox("Date / Period column", ["(none)"] + columns)
    target_col = st.sidebar.selectbox("Target (value) column", columns)

    # Prepare series
    if date_col != "(none)":
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
            series = df.set_index(date_col)[target_col].astype(float)
        except Exception as e:
            st.error(f"Could not parse date column: {e}")
            st.stop()
    else:
        series = df[target_col].astype(float).reset_index(drop=True)

    st.subheader("Time Series")
    st.line_chart(series)

    # ── Model parameters ─────────────────────────────────────────────────────
    st.sidebar.subheader("Holt-Winters Parameters")

    trend = st.sidebar.selectbox("Trend component", ["add", "mul", "None"], index=0)
    trend = None if trend == "None" else trend

    seasonal = st.sidebar.selectbox("Seasonal component", ["add", "mul", "None"], index=0)
    seasonal = None if seasonal == "None" else seasonal

    seasonal_periods = st.sidebar.number_input(
        "Seasonal periods",
        min_value=2,
        max_value=365,
        value=12,
        step=1,
        help="Number of time steps in a seasonal cycle (e.g. 12 for monthly data with yearly seasonality).",
    )

    forecast_steps = st.sidebar.number_input(
        "Forecast steps",
        min_value=1,
        max_value=500,
        value=12,
        step=1,
    )

    damped_trend = st.sidebar.checkbox("Damped trend", value=False)

    # ── Fit & forecast ────────────────────────────────────────────────────────
    if st.sidebar.button("Run Forecast"):
        try:
            model_kwargs = dict(
                trend=trend,
                damped_trend=damped_trend if trend is not None else False,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods if seasonal is not None else None,
            )
            model = ExponentialSmoothing(series, **model_kwargs)
            fit = model.fit(optimized=True)
            forecast = fit.forecast(forecast_steps)

            # Build result dataframe – align on common index to handle
            # initial observations that the model may leave as NaN
            fitted_values = fit.fittedvalues
            aligned = pd.DataFrame({"actual": series, "fitted": fitted_values}).dropna()
            actual_aligned = aligned["actual"]
            fitted_aligned = aligned["fitted"]
            result_df = pd.DataFrame({"Actual": series, "Fitted": fitted_values})
            forecast_df = pd.DataFrame({"Forecast": forecast})

            # ── Plot ──────────────────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(range(len(series)), series.values, label="Actual", color="steelblue")
            # Map aligned index positions for the fitted line
            fitted_positions = [series.index.get_loc(i) for i in actual_aligned.index]
            ax.plot(fitted_positions, fitted_aligned.values, label="Fitted", color="orange", linestyle="--")
            forecast_index = range(len(series), len(series) + forecast_steps)
            ax.plot(forecast_index, forecast.values, label="Forecast", color="green", linestyle="-.")
            ax.axvline(x=len(series) - 1, color="gray", linestyle=":", linewidth=1)
            ax.set_title("Holt-Winters Forecast")
            ax.set_xlabel("Time")
            ax.set_ylabel(target_col)
            ax.legend()
            st.subheader("Forecast Plot")
            st.pyplot(fig)

            # ── Metrics ───────────────────────────────────────────────────────
            mae = np.mean(np.abs(actual_aligned.values - fitted_aligned.values))
            rmse = np.sqrt(np.mean((actual_aligned.values - fitted_aligned.values) ** 2))
            mape_mask = actual_aligned.values != 0
            mape = np.mean(np.abs((actual_aligned.values[mape_mask] - fitted_aligned.values[mape_mask]) / actual_aligned.values[mape_mask])) * 100

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("AIC", f"{fit.aic:.2f}")
            col2.metric("MAE", f"{mae:.4f}")
            col3.metric("RMSE", f"{rmse:.4f}")
            col4.metric("MAPE", f"{mape:.2f}%")

            # ── Forecast table ────────────────────────────────────────────────
            st.subheader("Forecast Values")
            st.dataframe(forecast_df.rename(columns={"Forecast": target_col}), use_container_width=True)

            # ── Download ──────────────────────────────────────────────────────
            csv_buffer = BytesIO()
            forecast_df.to_csv(csv_buffer, index=True)
            st.download_button(
                label="⬇️ Download forecast as CSV",
                data=csv_buffer.getvalue(),
                file_name="hw_forecast.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Model fitting error: {e}")

else:
    st.info("👈 Please upload a CSV file using the sidebar to get started.")

    st.markdown(
        """
        ### How to use
        1. **Upload** a CSV file with at least one numeric column representing your time series.
        2. **Select** the date/period column (optional) and the target column.
        3. **Configure** the Holt-Winters parameters in the sidebar.
        4. Click **Run Forecast** to train the model and visualize results.

        ### About Holt-Winters
        The Holt-Winters method (Triple Exponential Smoothing) extends simple exponential smoothing
        to capture **trend** and **seasonality** in time-series data.  It is widely used for
        short- to medium-term forecasting of business and economic indicators.
        """
    )
