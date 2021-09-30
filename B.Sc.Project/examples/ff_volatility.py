# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:37:51 2020

This is an example of how to use armagarch package to use ARMA-GARCH module.
The module uses the data from Kenneth French's Data library and estimates
conditional volatility for excess market returns.

@author: Ian Khrashchevskyi
"""

import armagarch as ag
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# reading data
data = pd.read_excel('all_data_with_weather_linear_filled.xlsx')
data.index = pd.DatetimeIndex(data['date'])
# data2 = data[['PSI_S1','CO_S1','O3_S1','NO2_S1','SO2_S1','PM10_S1','PM2_5_S1']]
data2 = data[['PM10_S1']]
data2 = data2[:-33]

# define mean, vol and distribution
meanMdl = ag.ARMA(order = {'AR':1,'MA':0})
volMdl = ag.garch(order = {'p':1,'o':1,'q':1})
distMdl = ag.normalDist()

# create a model

model = ag.empModel(data2, meanMdl, volMdl, distMdl)
# fit model
model.fit()

# get the conditional mean
Ey = model.Ey
print("************************")
print(Ey)

# get conditional variance
ht = model.ht
cvol = np.sqrt(ht)
print("sdfsssssssssssssss")
print(cvol)
# get standardized residuals
stres = model.stres
print(stres)

frcst = model.predict(nsteps = 1)
print(frcst)

