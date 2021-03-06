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
its past values and also on the past values of other variables.
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
Akaike???s Information Criterion (AIC). Also, after applying the 
model, we check its compatibility with data. Using Durbin 
Watson test confirms this compatibility. Experimental results 
demonstrate the high efficiency of the proposed method in 
predicting air pollutants.  

## Train and Test Dataset  

The criteria used to measure the forecasting accuracy is root 
mean square error (RMSE). We will do one-step ahead forecasts 
for 200 times and every time we will add one row to the train 
dataset. It means that for example at first time, rows from 1 to 
992 are train dataset, 993th row is test dataset and we forecast 
993th row and calculate the error, then at second time rows from 
1 to 993 are train dataset and 994th row is test dataset then we 
forecast 994th row and calculate the error and etc. Rows from 1 
to 992 are always in the train dataset and every time one row is 
added to the train dataset.  

## Select Order of VAR Model  

To select proper order for VAR model, we use AIC criteria. 
We test orders from 1 to 100 and find AIC for each of them and 
then select the order which has the minimum AIC. In Fig bellow the 
value of AIC criteria for orders in range of 1 to 100 in shown. 
Minimum AIC occurs at the lag order of 6.  
![image](https://user-images.githubusercontent.com/44861408/135466048-a0ccd24f-1c90-4442-a08a-15e6f89f42f6.png)

## Evaluation of model  

The resulted RMSEs from applying VAR and Auto
Regression (AR) model are given in the Table bellow. In an AR 
model, we forecast one variable using a linear combination of its 
past values. It means that in AR model, we do not consider 
impact of variables on each other and each variable is forecasted 
only with its own past values. Also, in table, for every variable in the VAR model, we 
listed variables that are a cause for it. As you can see in the Table, the RMSEs of VAR model are less than AR model.  
![image](https://user-images.githubusercontent.com/44861408/135466471-a62f6c62-c8bf-4e2d-99f3-b028a5e6c48c.png)  

### actual and forecasted amount of SO2 and PM10 for 200 days :  
![image](https://user-images.githubusercontent.com/44861408/135467232-f43fd1a5-5d78-41c3-ad90-1459cf6beffb.png)  
  
![image](https://user-images.githubusercontent.com/44861408/135467325-9e86e2bb-7c54-4c0c-b489-69db8dcecc24.png)




## Check for Serial Correlation of Residuals  

After fitting our model, we used Durbin Watson test to check 
if there is any autocorrelation in the residuals. The results from 
the test are given in Table bellow. All the numbers are very close to 
number 2 or equal to it. This means that there is not any 
autocorrelation left in the residuals and so the compatibility 
between VAR model and our data is confirmed.  
![image](https://user-images.githubusercontent.com/44861408/135466751-2127d3c4-6d1e-47c8-a0af-1d2e6cf1c822.png)

## CONCLUSION  

Air pollution has become a serious concern in 21st century. 
Importance of forecasting air pollution is not ignorable. In this 
paper we tried to get use of correlation between different air 
pollutants and build a VAR model based on them. We used
Granger causality test to find causality relations between our 
variables for a better prediction. The study showed that because 
of impact of air pollutants on each other, applying multivariate 
approaches like VAR model has a good performance and can 
reduce forecasting errors.
Acknowledgments: This work is funded by Tehran Center
for Urban Statistics and Observatory (contract No. 250/46035).  
  
## published paper:  
this work was published in 2020 6th Iranian Conference on Signal Processing and Intelligent Systems (ICSPIS) and indexed in IEEE.  

### IEEE link:  
https://ieeexplore.ieee.org/document/9349617  

## References:  
[1] W. J. M. X. a. L. H. L., "Air Pollution Forecasts: An 
Overview," International Journal of Environmental 
Research and Public Health, pp. 154-780, 2018.  

[2] G. B. S. M. S. F. S. A.-M. I. P. F. C. L. L. G. S. &. F. F. 
Viegi, "Health effects of air pollution: a Southern 
European perspective," Chinese medical journal, 2020.   

[3] D. P. a. A. J. N. Tomar, "Air Quality Index Forecasting 
using Auto-regression Models," in IEEE International 
Students' Conference on Electrical,Electronics and 
Computer Science (SCEECS), Bhopal, India, 2020.   

[4] U. &. J. V. Kumar, "ARIMA forecasting of ambient air 
pollutants (O3, NO, NO2 and CO)," Stochastic 
Environmental Research and Risk Assessment, 2010.   

[5] J. J. S. &. N. N. Qiu, " Multivariate time series analysis 
from a Bayesian machine learning perspective," Ann 
Math Artif Intell, p. 1061???1082, 2020.   

[6] C. Sims, "Macroeconomics and Reality," Econometrica, 
pp. 1-48, 1980.   

[7] H. Wai Lun, "Application Of Vector Autoregressive 
Models To Hong Kong Air Pollution Data," september 


[8] T. A. G. D. S. B. Abhilash M.S.K., "Time Series 
Analysis of Air Pollution in Bengaluru Using ARIMA 
Model," springer, vol. 696, 2018.   

[9] C. G. Kwang-Jing Yii, "The Nexus between Technology 
Innovation and CO2 Emissions in Malaysia: Evidence 
from Granger Causality Test," Energy Procedia, vol. 
105, pp. 3118-3124, 2017.   

[10] C. I. C. P. A. &. R. N. Pardo Martinez, "An analysis for 
new institutionality in science, technology and 
innovation in Colombia using a structural vector 
autoregression model," European Research Studies 
Journal, vol. 22, no. 2, pp. 218-228, 2019.   

[11] S. Portet, "A primer on model selection using the 
Akaike Information Criterion," Infectious Disease 
Modelling, vol. 5, pp. 111-128, 2020.   

[12] B. &. B. J. Born, "Testing for Serial Correlation in 
Fixed-Effects Panel Data Models," Econometric 
Reviews, 2015.



