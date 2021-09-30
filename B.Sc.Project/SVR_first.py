from sklearn.svm import SVR
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# load the dataset
series = pd.read_excel('all_data_with_weather_filled.xlsx')
series= series[:-25]
series.index = pd.DatetimeIndex(series['date'])
# data_use = series[['NO2_S1','SO2_S1','CO_S1']]
# data_use = series[['NO2','PM2_5','CO','PSI','PM10','SO2']]
data_use = series[[  'CO_S1', 'O3_8_S1', 'O3_S1', 'NO2_S1', 'SO2_S1', 'PM10_S1','PSI_S1']]
series['date'] = series['date'].map(mdates.date2num)

# data_use = data_use[:-25]
dates = series['date'].values
prices = data_use['PM10_S1'].values

#Convert to 1d Vector
dates = np.reshape(dates, (len(dates), 1))
prices = np.reshape(prices, (len(prices), 1))
# print(dates)
# print(prices)


# svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1)
svr_rbf = SVR(kernel= 'linear',C= 1e3, gamma= 0.1)
svr_rbf.fit(dates[:-1], prices[:-1])
predicted =svr_rbf.predict(dates[-1:])
print(predicted)
print(prices[-1:])

def FindingErrors():
    predictions=[]
    for nobs in range(5, 1, -1):
        svr_rbf.fit(dates[:-nobs], prices[:-nobs])
        predicted = svr_rbf.predict(dates[-nobs:-nobs+1])
        predictions.append(predicted)
        print(predicted)
        print(prices[-nobs:-nobs+1])
        print(nobs)
        print("***********************************************************************")
    mape = np.mean(np.abs(predictions - prices[-5:-1])/np.abs(prices[-5:-1]))
    print(mape)
# FindingErrors()