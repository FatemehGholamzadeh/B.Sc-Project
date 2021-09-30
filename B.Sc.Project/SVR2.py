from sklearn.svm import SVR
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

# load the dataset
series = pd.read_excel('all_data_with_weather_filled.xlsx')
series= series[:-32]
series.index = pd.DatetimeIndex(series['date'])
data_use = series[['O3_8_S5', 'O3_S5', 'O3_1_S5', 'O3_8_S8', 'O3_S8', 'O3_8_S41', 'O3_8_S20']
]
# data_use = series[['PSI_S1','CO_S1','O3_1_S1']]

# data_use = series.drop(['date'],axis=1)

X = data_use.drop(['O3_8_S5'], axis = 1)
y = data_use['O3_8_S5']

# X_train = X[:-1]
# y_train = y[:-1]
# X_test = X[-1:]
# y_test = y[-1:]
#

Svr=SVR(kernel='linear', C=1)
# Svr.fit(X_train,y_train)
# # print(Svr.score(X_train,y_train))
# prediction = Svr.predict(X_test)
# print(prediction)
# print(y_test)
# error = np.sqrt(mean_squared_error(y_test,prediction)) #calculate rmse
# print('RMSE value of the SVR Model is:', error)

def FindingErrors():
    # predictions=[]
    nobsNum = 2
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
        # predictions.append(predicted)
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




# Set the parameters by cross-validation
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                      'C': [1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

# for c in [1, 10, 100, 1000]:
#     clf = SVR(kernel='linear',C=c)
#     clf.fit(X_train, y_train)
#     confidence = clf.score(X_train, y_train)
#     print(c,confidence)

# for k in ['linear','poly','rbf','sigmoid']:
#     clf = SVR(kernel=k)
#     clf.fit(X_train, y_train)
#     confidence = clf.score(X_train, y_train)
#     print(k,confidence)
