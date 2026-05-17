import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import root_mean_squared_error
import pmdarima as pm
from pmdarima.arima import auto_arima

"""
TASK 3:
Decomposition and ARIMA modeling with statsmodels (and pmdarima for auto-tuning).
version-1
OBJECTIVE:
Forecast wine sales using classical time series techniques.
1. Load and clean (translate columns from chinese to english).
2. Decomposition (trend, seasonality, residual).
3. Automatic SARIMA fitting.
4. Forecast 12 months ahead.
5. Evaluation with RMSE.
"""

def load_and_prepare_data(filepath):
    """
    Loads the Excel file, translates columns, and aggregates by date to create
    a time series.
    """

    print(f"Processing file: {filepath}")

    # Load the series with pandas
    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    # Clean column names
    df.columns = df.columns.str.strip()
    print("Columns before rename: ", df.columns.tolist())
    
    # Column translation
    column_map = {
        "ID": "id",
        "日期": "date",
        "商品名称": "product_name",
        "价格": "price", 
        "近30天销量（瓶）": "sales",
        "评论数": "reviews",
        "品牌": "brand",
        "被搜索次数": "searches"
    }

    df.rename(columns=column_map, inplace=True)
    print("Columns after rename: ", df.columns.tolist())

    # Convert date from integer format (YYYYMM) to datetime
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m")
    
    # Aggregate by date to get TOTAL sales per month
    ts_data = df.groupby("date")["sales"].sum().reset_index()
    
    # Ensure index is datetime
    ts_data.set_index("date", inplace=True)
    
    # Set monthly frecuency (MS = Month Start)
    ts_data = ts_data.asfreq('MS').fillna(0)

    print(f"Time series generated. Total months: {len(ts_data)}")
    return ts_data

def classical_decomposition(ts_data):
    """
    Performs classical decomposition (trend, seasonality, residual).
    If there are not enough observations (minimum 24 months), it will
    skkip and log the limitation.
    """
    print("\nPerforming classical decomposition...")
    
    try:
        result = seasonal_decompose(ts_data, model="additive") 

        # Plot the decomposition
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        result.observed.plot(ax=ax1, title="Observed")
        result.trend.plot(ax=ax2, title="Trend")
        result.seasonal.plot(ax=ax3, title="Seasonality")
        result.resid.plot(ax=ax4, title="Residual")
        plt.tight_layout()
        plt.savefig("/results/decomposition_analysis.png")
        print("Decomposition plot saved as 'decomposition_analysis.png'")

        return result
    except ValueError as e:
        print(f"Decomposition could not be performed: {e}")
        print("Explanation: At least 24 monthly observations are required.")
        return None

def fit_and_forecast_sarima(ts_data):
    """
    Fits a SARIMA model with automatic parameter selection.
    Forecast the next 12 months.
    Evaluates the model using RMSE.
    """
    print("\nStarting SARIMA modeling")

    # Split into training and test
    n_test = 3 
    if(len(ts_data) <= n_test):
        print(f"Warning: Only {len(ts_data)} months available."
              "Using all data for training, forecast may be unreliable.")
        train = ts_data
        test = ts_data
    else:
        train = ts_data.iloc[:-n_test]
        test = ts_data.iloc[-n_test:]

    print(f"Training: {len(train)} months | Test: {len(test)} months")

    # Auto-ARIMA: find best parameters
    print("Searching for best SARIMA parameters...")
    auto_model = auto_arima(
        train,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        trace=True
    )

    print(f"\nBest model found:\n{auto_model.summary()}")

    forecast, conf_int = auto_model.predict(n_periods=n_test, return_conf_int=True)
    forecast_index = pd.date_range(start=train.index[-1] + pd.offsets.MonthBegin(), periods=n_test, freq="MS")
    forecast_series = pd.Series(forecast, index=forecast_index)
    
    if(len(test) == n_test):
        rmse = root_mean_squared_error(test, forecast_series)
        print(f"\nEvaluation results:\nRMSE: {rmse:.2f}")
    else:
        rmse = None
        print("\nEvaluation skipped: not enough test data.")
        
    plt.figure(figsize=(14, 7))
    plt.plot(train.index, train, label="Training (Historical)")
    if(len(test) == n_test):
        plt.plot(test.index, test, label="Actual Data (Test)", color="green")
    plt.plot(forecast_series.index, forecast_series, label="SARIMA Forecast", color="red", linestyle="--")
    plt.fill_between(forecast_series.index, conf_int[:, 0], conf_int[:, 1], color="pink", alpha=0.3, label="95% Confidence Interval")
    plt.title(f"Wine Sales Forecast" + (f" (RMSE: {rmse:.2f})" if rmse else ""))
    plt.legend()
    plt.grid(True)
    plt.savefig("results/sarima_forecast.png")
    print("Forecast plot saved as 'sarima_forecast.png'")

if __name__ == "__main__":
    FILE_WINE_SALES = "data/wine_sales.xlsx"

    # Execution flow
    wine_sales = load_and_prepare_data(FILE_WINE_SALES)
    
    if(wine_sales is not None):
        classical_decomposition(wine_sales["sales"])
        fit_and_forecast_sarima(wine_sales["sales"])
