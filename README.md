# HW_forecast
Holt Winters Forecast - Promidat - JC _ Jose Sirias Dev

A **Streamlit** web application for time-series forecasting using the **Holt-Winters Exponential Smoothing** method.

## Features

- Upload any CSV file with time-series data
- Optional date/period column parsing
- Choose trend and seasonal components (`additive`, `multiplicative`, or `None`)
- Configure seasonal periods and forecast horizon
- Optional damped trend
- Interactive forecast plot
- Accuracy metrics: AIC, MAE, RMSE, MAPE
- Download forecast results as CSV

## Requirements

- Python 3.9+

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`) in your browser.

1. **Upload** a CSV file using the sidebar.
2. **Select** the date column (optional) and the target numeric column.
3. **Configure** Holt-Winters parameters.
4. Click **Run Forecast**.
