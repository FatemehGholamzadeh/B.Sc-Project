# Import libraries
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.varmax import VARMAX
from random import random


# load the dataset
series = pd.read_excel('all_data_with_weather_filled.xlsx')
series.index = pd.DatetimeIndex(series['date'])
data_use = series[[ 'PSI_S1', 'CO_S1','NO2_S1','SO2_S1','PM10_S1', 'PM2_5_S1']]
data_use = data_use[1550:-25]
exog = series[['WindSpeed10', 'Temperture', 'Pressure']]
exog = exog[1550:-25]
# print(data_use)

# # fit model
# model = VARMAX(data_use.values, exog=exog.values, order=(1, 1))
# model_fit = model.fit(disp=False)
# print(model_fit.summary())
#
# # make prediction
# data_exog2 = [[3.25,16.46,1013]]
# yhat = model_fit.forecast(exog=data_exog2)
# print(yhat)


def FindingErrors(data_use,exog):
    forecast = pd.DataFrame(columns=data_use.columns )
    for nobs in range(150, 1, -1):
        df_train, df_test = data_use[0:-nobs], data_use[-nobs:-nobs + 1]
        model = VARMAX(data_use.values, exog=exog.values, order=(1, 1))
        model_fit = model.fit(disp=False)
        # print(model_fitted.summary())
        data_exog2=exog[-nobs:-nobs + 1]
        fc = model_fit.forecast(exog=data_exog2)
        df_forecast = pd.DataFrame(fc, index=data_use.index[-nobs:-nobs + 1], columns=data_use.columns )
        print(df_forecast)
        frames = [ forecast ,df_forecast]
        forecast = pd.concat(frames)
    return forecast

# FindingErrors(data_use,exog)



def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    rmse = np.mean((forecast - actual) ** 2) ** .5
    # a = forecast - actual
    # cols = actual.columns
    # m = 0
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
    return mape,rmse


mape,rmse = forecast_accuracy(FindingErrors(data_use,exog),data_use[-150:-1])
# print(a)
print(mape)
print(rmse)