from cmath import sqrt

import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR


# reading data
data = pd.read_excel('s14_linear.xlsx')
data.index = pd.DatetimeIndex(data['date'])
data_use = data[['CO','NO2','SO2','PM10','PM2_5']]


#creating the train and validation set
train = data_use[:int(0.8*(len(data_use)))]
valid = data_use[int(0.8*(len(data_use))):]


model = VAR(endog=train)
model_fit = model.fit()
# print(model_fit.summary())

# make prediction on validation
prediction = model_fit.forecast(model_fit.y, steps=len(valid))
print(prediction)
print(data_use[-6:])

cols = data_use.columns
#converting predictions to dataframe
# pred = pd.DataFrame(index=range(0,len(prediction)),columns=[cols])
# for j in range(0,13):
#     for i in range(0, len(prediction)):
#        pred.iloc[i][j] = prediction[i][j]

# #check rmse
# for i in cols:
#     print('rmse value for', i, 'is : ', sqrt(mean_squared_error(pred[i], valid[i])))


#make final predictions
# model = VAR(endog=data)
# model_fit = model.fit()
# yhat = model_fit.forecast(model_fit.y, steps=1)
# print(yhat)