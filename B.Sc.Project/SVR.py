from sklearn.svm import SVR
import pandas as pd
import numpy as np

# load the dataset
series = pd.read_excel('all_data_with_weather_filled.xlsx')
series= series[:-32]
series.index = pd.DatetimeIndex(series['date'])
data_use = series[['O3_8_S5', 'O3_S5', 'O3_1_S5', 'O3_8_S8', 'O3_S8', 'O3_8_S41', 'O3_8_S20']]

X = data_use.drop(['O3_8_S5'], axis = 1)
y = data_use['O3_8_S5']

Svr=SVR(kernel='linear', C=1)

def FindingErrors():
    nobsNum = 100
    forecast = pd.DataFrame()
    for nobs in range(nobsNum, 1, -1):
        X_train = X[:-nobs]
        y_train = y[:-nobs]
        X_test = X[-nobs:-nobs+1]
        y_test = y[-nobs:-nobs+1]
        Svr.fit(X_train, y_train)
        predicted = Svr.predict(X_test)
        df_forecast = pd.DataFrame(predicted, index=data_use.index[-nobs:-nobs + 1], columns=['O3_8_S5'])
        frames = [forecast, df_forecast]
        forecast = pd.concat(frames)
        print(predicted)
        print(y_test)
        print(nobs)
        print("***********************************************************************")
    actual =y[-nobsNum:-1].to_frame()
    mape = np.mean(np.abs(forecast -actual)/np.abs(actual))
    rmse = np.mean((forecast - actual) ** 2) ** .5
    print(mape)
    print(rmse)

FindingErrors()
