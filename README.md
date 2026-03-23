# Ex.No: 6               HOLT WINTERS METHOD
### Date: 23/02/2026
### Reg No:212224240053


### AIM:
To run program on holt winters method

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:

```

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error




data = pd.read_csv('/content/INDIA VIX_minute.csv', parse_dates=['date'],index_col='date')

data.head()



# Ensure index is datetime
data.index = pd.to_datetime(data.index, errors='coerce')

# Drop rows where datetime parsing failed (if any)
data = data.dropna()

# Now resample monthly (month start)
data_monthly = data.resample('MS').sum()

print(data_monthly.head())
data_monthly.plot()
plt.show()




scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_monthly['close'].values.reshape(-1, 1)).flatten(),
    index=data_monthly.index
)

scaled_data.plot()
plt.show()



import numpy as np
from sklearn.metrics import mean_squared_error

# Shifted scaled_data by +1 already
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul').fit()

# Forecast with aligned index
test_predictions_add = pd.Series(
    model_add.forecast(steps=len(test_data)),
    index=test_data.index
)

# Plot
ax = train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add","test_data"])
ax.set_title('Visual evaluation')

# RMSE
print("RMSE:", np.sqrt(mean_squared_error(test_data, test_predictions_add)))

# Variance and mean
print("Scaled variance sqrt, mean:", np.sqrt(scaled_data.var()), scaled_data.mean())





final_model = ExponentialSmoothing(
    data_monthly['close'],  # pick one column
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

# Forecast for next year
final_predictions = final_model.forecast(steps=12)

# Plot
ax = data_monthly['close'].plot(label="data_monthly")
final_predictions.plot(ax=ax, label="final_predictions")
ax.legend()
ax.set_xlabel('Date')
ax.set_ylabel('Close Price')
ax.set_title('Prediction')





```


### OUTPUT:



<img width="1155" height="733" alt="image" src="https://github.com/user-attachments/assets/61e3b33e-1cd1-4368-802d-ed096a0cfb8d" />


<img width="1171" height="778" alt="image" src="https://github.com/user-attachments/assets/a9791538-3fe5-4082-ae2d-5b9de4ae8e4a" />


<img width="1349" height="610" alt="image" src="https://github.com/user-attachments/assets/a3d275b6-2d90-4524-b53a-8dee5affdd0c" />


<img width="1370" height="613" alt="image" src="https://github.com/user-attachments/assets/d9f6a3da-f0a2-4671-9526-45aa3f93c6d5" />




### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
