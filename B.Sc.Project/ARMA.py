import pandas as pd
from statsmodels.tsa.stattools import ARMA
from arch import arch_model
import numpy as np
from  matplotlib import pylab
from statsmodels.tsa.stattools import ARMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import het_arch
import pmdarima
from pmdarima.arima import ADFTest
import arch


# reading data
data = pd.read_excel('all_data_with_weather_linear_filled.xlsx')
data.index = pd.DatetimeIndex(data['date'])
# data_use = data[['PSI_S1','CO_S1','NO2_S1','SO2_S1','PM10_S1','PM2_5_S1']]
# data_use = data[['PSI_S1','CO_S1','NO2_S1','SO2_S1','PM10_S1','PM2_5_S1']]
data_use = data[['PM10_S1']]
data_use = data_use[:-32]


# ADF Rest
# adf = ADFTest(alpha=0.05)
# print(adf.should_diff(data_use))


# fit ARIMA on returns
# arima_model= pmdarima.auto_arima(data_use,start_p=0,d=1,start_q=0,
#                                  max_p=5 , max_d=5, max_q=5 , start_P=0,
#                                  D=1 , start_Q=0 , max_P=5 , max_D= 5 ,
#                                  max_Q= 5 , m=12 , seasonal=True ,
#                                  error_action='warn' , trace= True ,
#                                  suppress_warnings=True , stepwise=True,
#                                  random_state=20 , n_fits=50
#                                  )
# p, d, q = arima_model.order
# print(p,d,q)
# arima_model =  pmdarima.auto_arima(data_use)
# arima_residuals = arima_model.arima_res_.resid

# fit a GARCH(1,1) model on the residuals of the ARIMA model
# garch = arch.arch_model(arima_residuals, p=1, q=2)
# garch_fitted = garch.fit()

# Use ARIMA to predict mu
# mu = arima_model.predict(n_periods=1)
# predicted_mu = mu[0]
# print(predicted_mu)
# Use GARCH to predict the residual
# garch_forecast = garch_fitted.forecast(horizon=1)
# et = garch_forecast.mean['h.1']
# predicted_et = et.iloc[-1]

# Combine both models' output: yt = mu + et
# prediction = predicted_mu + predicted_et
# print((prediction))






# def FindingErrors(data_use):
#     forecast = pd.DataFrame(columns=data_use.columns)
#     for nobs in range(5, 1, -1):
#         df_train, df_test = data_use[0:-nobs], data_use[-nobs:-nobs + 1]
#
#         # fit ARIMA on returns
#         arima_model = pmdarima.auto_arima(df_train)
#         arima_residuals = arima_model.arima_res_.resid
#
#         # fit a GARCH(1,1) model on the residuals of the ARIMA model
#         garch = arch.arch_model(arima_residuals, p=1, q=2)
#         garch_fitted = garch.fit()
#
#         # Use ARIMA to predict mu
#         mu = arima_model.predict(n_periods=1)
#         predicted_mu = mu[0]
#
#         # Use GARCH to predict the residual
#         garch_forecast = garch_fitted.forecast(horizon=1)
#         et = garch_forecast.mean['h.1']
#         predicted_et = et.iloc[-1]
#
#         # Combine both models' output: yt = mu + et
#         prediction = predicted_mu + predicted_et
#         df_forecast = pd.DataFrame(prediction, index=data_use.index[-nobs:-nobs + 1], columns=data_use.columns)
#         frames = [forecast, df_forecast]
#         forecast = pd.concat(frames)
#     return forecast
#
# def forecast_accuracy(forecast, actual):
#     mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
#     return mape
# #
# # print((FindingErrors(data_use)))
# # print(data_use[-5:-1])
# print(forecast_accuracy(FindingErrors(data_use),data_use[-20:-1]))