# B.Sc.Project
Air Pollution Forecasting for Tehran City using Vector Auto Regression - B.Sc. Project - Fall 2020  
## Abstract   
Recently, in many urban areas, air quality has 
decreased because of human activities such as biomass burning 
and development of industrialization. There are some harmful 
pollutants like CO, NO2, and SO3 in the atmosphere that can lead 
to diseases such as asthma and lung cancer. One of the cities that 
seriously faces the problem of air pollution is Tehran. So, air 
pollution prediction for this city is of great importance and may 
lead to provide proper actions and controlling strategies. In this 
paper, we are going to use Vector Auto Regression (VAR) model 
to forecast daily concentrations of air pollutions in Tehran city for 
one day ahead. Since there are some correlations between air 
pollutants, it seems to get better results in forecasting if we get use
of these correlations. So our approach is to use a VAR model which 
considers the impact of variables on each other. For choosing the 
right variables in our model, we will use a causality test.
Experimental results demonstrate the high efficiency of the 
proposed approach in forecasting the concentrations of air 
pollutions in Tehran city.
    
## Introduction 

Our dataset contains the concentrations of different air 
pollutants from several stations in the city of Tehran. By 
performing some tests such as causality test, it was deduced that 
pollutants of a given station have some impact on each other and 
also on pollutants of nearby stations. So in this paper we are 
going to capture these correlations to improve the accuracy of 
forecasting. For this purpose, multivariate approach is 
recommended.  
  
A multivariate time series is a time series which has more 
than one time-independent variable. Each variable depends on 
its past values and also on the past values of other variables [5].
Vector auto regression (VAR) model is a method for predicting 
such time series.
Our dataset includes recorded amounts of pollutants like PSI
(Pollutant Standards Index), CO, O3, NO2, SO2, PM10
(particulate matter with a diameter of 10 micrometers or less), 
PM2_5 (particulate matter with a diameter of 2.5 micrometers or 
less) and also some geographical factors like temperature, wind 
speed, wind degree and relative humidity. We also should apply 
some preprocessing on data to deal with missing values. Before 
modeling the data with VAR models, some tests should be done 
on the dataset. First of all, the stationarity of data should be 
checked. To this end, we use unit root test. Then, the causality 
of the variables (amount of different air pollutants) should be 
investigated. We use Granger causality test. 
For selecting the proper order of model, we used the 
Akaike’s Information Criterion (AIC). Also, after applying the 
model, we check its compatibility with data. Using Durbin 
Watson test confirms this compatibility. Experimental results 
demonstrate the high efficiency of the proposed method in 
predicting air pollutants.
