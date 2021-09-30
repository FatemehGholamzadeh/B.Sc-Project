import pandas as pd
from pmdarima.arima import auto_arima
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.ar_model import AutoReg

data = pd.read_excel('all_data_with_weather_linear_filled.xlsx')
data.index = pd.DatetimeIndex(data['date'])
data_use = data[['NO2_S1']]
data_use = data_use[:-25]

nobs = 10
df_train, df_test = data_use[0:-nobs], data_use[-nobs:-nobs + 1]
model = AutoReg(df_train,29)
model_fitted = model.fit()
# print(len(model_fitted.resid))
# print(model.)
print(acorr_ljungbox(model_fitted.resid, lags = 10))
# print(model_fitted.test_serial_correlation())