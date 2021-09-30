import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import VAR,SVAR
import xlrd
import matplotlib.pyplot as plot
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf
from Tests import grangers_causation_matrix
from Tests import cointegration_test
from Tests import adfuller_test
from Tests import forecast_accuracy
from itertools import chain, combinations
# from AR import makeDataSet



# reading data
# data = pd.read_excel('s14_linear.xlsx')
data = pd.read_excel('all_data_with_weather_filled.xlsx')
data.index = pd.DatetimeIndex(data['date'])

data2 = data[[ 'PSI_S7', 'CO_S7', 'O3_1_S7', 'O3_8_S7', 'O3_S7', 'NO2_S7', 'SO2_S7', 'PM10_S7', 'PM2_5_S7', 'WindSpeed10', 'Temperture', 'Pressure']]


# data_use = makeDataSet(data2)
# list = []
# for col in data.columns:
#     list.append(col)
# list.pop(0)
# print(list)
# data_use = data[list]
# print(data_use)
# data.__delitem__('date')
data_use = data2[1550:-25]
# data_use = data[['O3_S1']]
# data_use = data_use[:-32]


# test causality
# maxlag=12
# test = 'ssr-chi2test'
# print(grangers_causation_matrix(data_use, variables = data_use.columns))
df = grangers_causation_matrix(data_use, variables = data_use.columns)
df.to_csv('out.csv')


# cointegration_test
# print(cointegration_test(data_use))


# making test and train dataset
# nobs = 200
# df_train, df_test = data_use[0:-nobs], data_use[-nobs:-nobs+1]


# ADF Test on each column
# for name, column in df_train.iteritems():
#     adfuller_test(column, name=column.name)
#     print('\n')


# # select the right lag
def AICplot(df_train):
    model = VAR(df_train)
    aics = []
    indexes = []
    for i in range(0, 100, 5):
        indexes.append(i)
    for i in range(1, 100):
        result = model.fit(i)
        print('Lag Order =', i)
        aics.append(result.aic)
        print('AIC : ', result.aic)
    print(min(aics))
    plot.figure(figsize=(10, 5))
    plot.plot(aics, 'g-o', color='green')
    # plot.ylim(0,60)
    plot.xlim(0, 99)
    # plot.style.use('ggplot')
    plot.xticks(indexes)
    plot.yticks([28, 29, 30, 31, 32])
    plot.title('Finding Proper Order Of VAR model', fontsize=18)
    plot.ylabel('AIC', fontsize=15)
    plot.xlabel('order of model', fontsize=15)
    plot.show()

# AICplot(df_train)
# model = VAR(df_train)
# x = model.select_order(maxlags=301)
# print(x.summary())

# model_fitted = model.fit(6)
# print(model_fitted.summary())

# def adjust(val, length= 6): return str(val).ljust(length)


# Check for Serial Correlation of Residuals (Errors) using Durbin Watson Statistic
# out = durbin_watson(model_fitted.resid)
# for col, val in zip(data_use.columns, out):
#     print(adjust(col), ':', round(val, 2))


def FindingErrors():
    forecast = pd.DataFrame(columns=data_use.columns )
    for nobs in range(200, 1, -1):
        df_train, df_test = data_use[0:-nobs], data_use[-nobs:-nobs + 1]
        model = VAR(df_train)
        model_fitted = model.fit(6)
        # print(model_fitted.summary())
        forecast_input = df_train.values[-6:]
        fc = model_fitted.forecast(y=forecast_input, steps=1)
        df_forecast = pd.DataFrame(fc, index=data_use.index[-nobs:-nobs + 1], columns=data_use.columns )
        frames = [ forecast ,df_forecast]
        forecast = pd.concat(frames)
    return forecast


def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    rmse = np.mean((forecast - actual) ** 2) ** .5
    print(actual)
    print(forecast)
    a = forecast - actual
    cols = actual.columns
    m = 0
    # for col in cols:
    #     maxActual =actual[col].max()
    #     maxForecast = forecast[col].max()
    #     m = max(maxActual,maxForecast)
    #     plot.figure(figsize=(8, 4))
    #     ax = actual[col][-200:].plot(color='green')
    #     forecast[col][-200:].plot(ax=ax,color='tab:orange')
    #     ax.legend(["Actual","Forecast"]);
    #     plot.ylim(0, m + 10)
    #     plot.ylabel(col)
    #     plot.show()
    return mape,rmse,a


# mape,rmse,a = forecast_accuracy(FindingErrors(),data_use[-200:-1])
# print(a)
# print(mape)
# print(rmse)
