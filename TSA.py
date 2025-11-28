# i. Different forecasting techniques

# Simple Exponential Smoothing (SES)
ses_model = SimpleExpSmoothing(train).fit()
ses_pred = ses_model.forecast(len(test))

# Simple Moving Average (SMA)
window = 3
sma_pred = train.rolling(window=window).mean().iloc[-1]
sma_forecast = pd.Series([sma_pred] * len(test), index=test.index)

# Holt-Winters Smoothing
hw_model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=12).fit()
hw_pred = hw_model.forecast(len(test))

# ii. Calculate evaluation metrics
def calc_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}

print("SES Metrics:", calc_metrics(test, ses_pred))
print("SMA Metrics:", calc_metrics(test, sma_forecast))
print("HW Metrics:", calc_metrics(test, hw_pred))

# iii. Identify trends and seasonal patterns
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Actual', marker='o')
plt.plot(test.index, ses_pred, label='SES', marker='x')
plt.plot(test.index, hw_pred, label='Holt-Winters', marker='s')
plt.legend()
plt.show()




week 5

# ============================================================================
# LAB 5: WHITE NOISE AND STATIONARITY
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

# i. Generate white noise
white_noise = np.random.normal(0, 1, len(data))

# ii. Compare graphs
fig, axes = plt.subplots(2, 1, figsize=(12, 6))
axes[0].plot(white_noise)
axes[0].set_title('White Noise')
axes[1].plot(data)
axes[1].set_title('Time Series Data')
plt.show()

# iii. Statistical tests for stationarity
# Augmented Dickey-Fuller Test
adf_result = adfuller(data)
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")
print(f"Stationary: {adf_result[1] < 0.05}")

# KPSS Test
kpss_result = kpss(data)
print(f"
KPSS Statistic: {kpss_result[0]}")
print(f"p-value: {kpss_result[1]}")



week 6

 # ============================================================================
# LAB 6: TREND DETECTION AND ACF/PACF
# ============================================================================

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# i. Detect trends using moving averages
ma_trend = data.rolling(window=12).mean()
plt.plot(data, label='Original')
plt.plot(ma_trend, label='Trend (MA-12)', linewidth=2)
plt.show()

# ii. Plot ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(data.dropna(), lags=40, ax=axes[0])
plot_pacf(data.dropna(), lags=40, ax=axes[1])
plt.show()



week 7

#============================================================================
# LAB 7: AR MODEL
# ============================================================================

import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

# i. Examine ACF and PACF (already plotted above)
# PACF cuts off at lag p → AR(p) model

# ii. Fit AR(1) model
ar1_model = AutoReg(train, lags=1).fit()
ar1_pred = ar1_model.predict(start=len(train), end=len(train)+len(test)-1)
print(f"AR(1) AIC: {ar1_model.aic}")

# iii. Fit higher lag AR models
ar3_model = AutoReg(train, lags=3).fit()
ar3_pred = ar3_model.predict(start=len(train), end=len(train)+len(test)-1)
print(f"AR(3) AIC: {ar3_model.aic}")

plt.plot(test.values, label='Actual')
plt.plot(ar1_pred, label='AR(1)')
plt.plot(ar3_pred, label='AR(3)')
plt.legend()
plt.title('AR Model Comparison')
plt.show()



week 8
# ============================================================================
# LAB 8: MA MODEL
# ============================================================================

import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# i. Plot ACF and PACF (already done in Lab 6)

# ii. Fit MA(1) model
ma1_model = ARIMA(train, order=(0, 0, 1)).fit()
ma1_pred = ma1_model.forecast(steps=len(test))
print(f"MA(1) AIC: {ma1_model.aic}")

# iii. Fit higher lag MA model
ma3_model = ARIMA(train, order=(0, 0, 3)).fit()
ma3_pred = ma3_model.forecast(steps=len(test))
print(f"MA(3) AIC: {ma3_model.aic}")

# iv. Compare performances
print("MA(1) Metrics:", calc_metrics(test, ma1_pred))
print("MA(3) Metrics:", calc_metrics(test, ma3_pred))

plt.figure(figsize=(12, 4))
plt.plot(test.values, label='Actual')
plt.plot(ma1_pred.values, label='MA(1)')
plt.plot(ma3_pred.values, label='MA(3)')
plt.legend()
plt.title('MA Model Comparison')
plt.show()







week 9

# ============================================================================
# LAB 9: ARMA MODEL
# ============================================================================

import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# i. Initialize ARMA model
arma_model = ARIMA(train, order=(1, 0, 1))

# ii. Train the model
arma_fit = arma_model.fit()
print(f"ARMA(1,1) Summary:
{arma_fit.summary()}")

# iii. Generate forecasts
arma_pred = arma_fit.forecast(steps=len(test))
print("ARMA Metrics:", calc_metrics(test, arma_pred))

plt.plot(test.values, label='Actual')
plt.plot(arma_pred.values, label='ARMA(1,1)')
plt.legend()
plt.title('ARMA Model Forecast')
plt.show()



week 10

# ============================================================================
# LAB 10: ARIMA MODEL
# ============================================================================

import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# i. Initialize ARIMA model with p, d, q parameters
# p=1 (AR terms), d=1 (differencing), q=1 (MA terms)
arima_model = ARIMA(train, order=(1, 1, 1))

# ii. Train the model
arima_fit = arima_model.fit()
print(f"ARIMA(1,1,1) Summary:
{arima_fit.summary()}")

# iii. Generate forecasts
arima_pred = arima_fit.forecast(steps=len(test))
print("ARIMA Metrics:", calc_metrics(test, arima_pred))

plt.plot(test.values, label='Actual')
plt.plot(arima_pred.values, label='ARIMA(1,1,1)')
plt.legend()
plt.title('ARIMA Model Forecast')
plt.show()
