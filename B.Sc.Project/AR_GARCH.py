import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from arch import arch_model
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg, AR_DEPRECATION_WARN


# reading data
data = pd.read_excel('all_data_with_weather_linear_filled.xlsx')
data.index = pd.DatetimeIndex(data['date'])
# data_use = data[['PSI_S1','CO_S1','NO2_S1','SO2_S1','PM10_S1','PM2_5_S1']]
# data_use = data[['PSI_S1','CO_S1','NO2_S1','SO2_S1','PM10_S1','PM2_5_S1']]
data_use = data[['PM10_S1']]
data_use = data_use[:-32]


def FindingErrors(data_use):
    forecast = pd.DataFrame(columns=data_use.columns)
    for nobs in range(200, 1, -1):
        df_train, df_test = data_use[0:-nobs], data_use[-nobs:-nobs + 1]

        model = AutoReg(df_train, 29)
        model_fitted = model.fit()

        archm = arch_model(model_fitted.resid, p=1, o=1, q=1)
        arch_model_fitted = archm.fit()

        mu_pred = model_fitted.predict(start=len(df_train), end=len(df_train) + len(df_test) - 1)
        et_pred = arch_model_fitted.forecast(horizon=1).mean['h.1'].iloc[-1]

        # yt = mu + et
        prediction = mu_pred + et_pred

        df_forecast = pd.DataFrame(prediction, index=data_use.index[-nobs:-nobs + 1], columns=data_use.columns)
        frames = [forecast, df_forecast]
        forecast = pd.concat(frames)
    return forecast

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    return mape
#

# print(forecast_accuracy(FindingErrors(data_use),data_use[-200:-1]))


def double_exponential_smoothing(series, alpha, beta):
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result


def double_exp_denoising(data2):
    r = double_exponential_smoothing(data2.values, 0.8, 0.0001)
    r.pop(len(r) - 1)
    # # print(len(r))
    approximation = pd.DataFrame(r, columns=data2.columns, index=data2.index)
    forecast = FindingErrors(approximation)
    print(forecast)
    print(approximation[-200:-1])
    mape = forecast_accuracy(forecast, approximation[-200:-1])
    print(mape)
    plt.plot(approximation[-31:-1], label="actual")
    plt.plot(forecast[-30:], label="forecast")
    plt.plot(data2[-31:-1])
    plt.show()

double_exp_denoising(data_use)
