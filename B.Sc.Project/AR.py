import pandas as pd
import numpy as np
from scipy.ndimage import filters
from skimage.filters import gaussian
from statsmodels.tsa.ar_model import AutoReg, AR_DEPRECATION_WARN
from statsmodels.tsa.ar_model import AR
import matplotlib.pyplot as plt
import pywt
from skimage.restoration import (denoise_wavelet, estimate_sigma)


# reading data
data = pd.read_excel('s14_linear.xlsx')
data.index = pd.DatetimeIndex(data['date'])
# data2 = data[['PSI','CO','O3','NO2','SO2','PM10','PM2_5']]
data2 = data[['NO2']]

# data2 = data[['PM10']]
# data2 = data2[:-32]


def FindingErrors(data_use):
    forecast = pd.DataFrame(columns=data_use.columns)
    for nobs in range(200, 1, -1):
        df_train, df_test = data_use[0:-nobs], data_use[-nobs:-nobs + 1]
        # model = AR(df_train)
        # model_fitted = model.fit(6)
        # predictions = model_fitted.predict(start=len(df_train), end=len(df_train) + len(df_test) - 1, dynamic=False)
        model = AutoReg(df_train,29)
        model_fitted = model.fit()
        predictions = model_fitted.predict(start=len(df_train), end=len(df_train) + len(df_test) - 1)
        df_forecast = pd.DataFrame(predictions, index=data_use.index[-nobs:-nobs + 1], columns=data_use.columns)
        frames = [forecast, df_forecast]
        forecast = pd.concat(frames)
    return forecast


def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    rmse = np.mean((forecast - actual) ** 2) ** .5
    return mape,rmse


def mapeFinder(data2):
    mapes=[]
    rmses=[]
    for col in data2.columns:
        my_data = data2[[col]]
        mape ,rmse= forecast_accuracy(FindingErrors(my_data), my_data[-200:-1])
        mapes.append(mape)
        rmses.append(rmse)
    return mapes,rmses


def waveletDenoising():
    # Decompose into wavelet components, to the level selected:
    # cA, cD = pywt.dwt(data2.values, 'db1','smooth')
    w = pywt.Wavelet('haar')
    cA, cD = pywt.dwt(data2.values, wavelet=w, mode='constant')
    approximation = pd.DataFrame(cA, columns=data2.columns, index=data2.index)
    forecast = FindingErrors(approximation)
    mape = forecast_accuracy(forecast, approximation[-200:-1])
    print(mape)
    plt.plot(approximation[-31:-1], label="actual")
    plt.plot(forecast[-30:], label="forecast")
    plt.plot(data2[-31:-1])
    plt.show()

# waveletDenoising()

# ***** Find all MAPEs *****

# print(mapeFinder(data2))
# print(FindingErrors(data2))
# print(data2[-200:-1])

def exponential_smoothing(series, alpha):
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result


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


def plotDoubleExponentialSmoothing(series, alphas, betas):
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(20, 8))
        plt.plot(series[-50:], label="Actual")
        for alpha in alphas:
            for beta in betas:
                plt.plot(double_exponential_smoothing(series, alpha, beta)[-51:-1],
                         label="Alpha {}, beta {}".format(alpha, beta))

        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Double Exponential Smoothing")
        plt.grid(True)
        plt.show()

# plotDoubleExponentialSmoothing(data2.values , alphas=[0.8], betas=[0.0001])

# r = exponential_smoothing(data2.values,0.9)

def exp_denoising():
    r = exponential_smoothing(data2.values,0.7)
    approximation = pd.DataFrame(r, columns=data2.columns, index=data2.index)
    forecast = FindingErrors(approximation)
    # print(forecast)
    # print(approximation[-400:-1])
    mape,rmse = forecast_accuracy(forecast, approximation[-200:-1])
    print(mape)
    print(rmse)
    plt.plot(approximation[-51:-1], label="actual")
    plt.ylabel(data2.columns)
    plt.plot(data2[-51:-1])
    # plt.plot(forecast[-50:], label="forecast")
    plt.legend(["smoothed", "actual"])
    plt.show()

exp_denoising()

def makeDataSet(data2):
    approximation = pd.DataFrame(columns=data2.columns, index=data2.index)
    for col in data2.columns:
        r = exponential_smoothing(data2[col].values, 0.7)
        approximation[col] = r
    return approximation

# print(makeDataSet(data2))
# print("******************")
# print(data2)



def double_exp_denoising():
    r = double_exponential_smoothing(data2.values, 0.4, 0.0001)
    r.pop(len(r) - 1)
    # # print(len(r))
    approximation = pd.DataFrame(r, columns=data2.columns, index=data2.index)
    forecast = FindingErrors(approximation)
    # print(forecast)
    # print(approximation[-400:-1])
    mape,rsme = forecast_accuracy(forecast, approximation[-400:-1])
    print(rsme)
    print(mape)
    plt.plot(approximation[-201:-1])
    print(approximation[-201:-1])
    # plt.plot(forecast[-100:], label="forecast")
    plt.plot(data2[-201:-1])
    plt.ylabel(data2.columns)
    plt.legend(["smoothed","actual"])
    print(data2[-201:-1])
    plt.show()

# print(forecast_accuracy(FindingErrors(data2),data2[-200:-1]))
# double_exp_denoising()