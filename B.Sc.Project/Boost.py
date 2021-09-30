import pandas as pd
import numpy as np
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor, plot_importance
from matplotlib import pyplot

# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols = list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))

    # put it all together
    agg = concat(cols, axis=1)

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]


# fit an xgboost model and make a one step prediction
def xgboost_forecast(train, testX):
    # transform list into array
    train = asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(trainX, trainy)

    # make a one-step prediction
    yhat = model.predict(asarray([testX]))
    return yhat[0]


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    print(train)
    print("*************************")
    print(test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        testX, testy = test[i, :-1], test[i, -1]
        # fit model on history and make a prediction
        yhat = xgboost_forecast(history, testX)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        # summarize progress
        print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
    # estimate prediction error
    mape = np.mean(np.abs(predictions - test[:, -1]) / np.abs(test[:, -1]))
    rmse = np.mean((predictions - test[:, -1]) ** 2) ** .5
    # error = mean_absolute_error(test[:, -1], predictions)
    return mape,rmse,test[:, -1], predictions


# load the dataset
series = pd.read_excel('all_data_with_weather_filled.xlsx')
series.index = pd.DatetimeIndex(series['date'])
data_use = series[[  'CO_S1', 'O3_8_S1', 'O3_S1', 'NO2_S1', 'SO2_S1', 'PM2_5_S1','PSI_S1']]

data_use = data_use[:7]
values = data_use.values
# transform the time series data into supervised learning
data = series_to_supervised(values, n_in=1,n_out=1)
print(data)
# for a in data:
#     print(a)
#     print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

# evaluate
# mape,rmse, y, yhat = walk_forward_validation(data, 2)
# print('MAPE: %.3f' % mape)
# print('RMSE: %.3f' % rmse)

# plot expected vs preducted
# pyplot.plot(y, label='Expected')
# pyplot.plot(yhat, label='Predicted')
# pyplot.legend()
# pyplot.show()
