
---

### `sales_forecasting.py`

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import os

# Create directories for results
os.makedirs("results", exist_ok=True)

# Load the dataset
data_path = "data/sales_data.csv"
try:
    sales_data = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")
except FileNotFoundError:
    print(f"Dataset not found. Please ensure '{data_path}' exists.")
    exit()

# Data Preprocessing
sales_data = sales_data.sort_index()
sales_data = sales_data.fillna(method="ffill")  # Fill missing values
sales_data["Sales"] = sales_data["Sales"].replace([np.inf, -np.inf], np.nan).dropna()

# Exploratory Analysis
plt.figure(figsize=(12, 6))
sns.lineplot(data=sales_data, x=sales_data.index, y="Sales", label="Sales")
plt.title("Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.savefig("results/sales_trend.png")
plt.show()

# ARIMA Modeling
model = ARIMA(sales_data["Sales"], order=(5, 1, 0))
fitted_model = model.fit()

# Forecasting
forecast_steps = 30  # Predict next 30 days
forecast = fitted_model.forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=sales_data.index[-1], periods=forecast_steps + 1, freq="D")[1:]

# Plot Forecast
plt.figure(figsize=(12, 6))
plt.plot(sales_data.index, sales_data["Sales"], label="Historical Sales")
plt.plot(forecast_index, forecast, label="Forecast", linestyle="--", color="red")
plt.title("Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.savefig("results/forecast_plot.png")
plt.show()

# Save Summary Statistics
summary = sales_data.describe()
with open("results/summary_stats.txt", "w") as f:
    f.write("Sales Data Summary Statistics\n")
    f.write(summary.to_string())

print("Analysis complete. Results saved in the 'results/' folder.")
