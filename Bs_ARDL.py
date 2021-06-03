# -*- coding: utf-8 -*-
"""
Created on Fri May 14 12:52:38 2021

@author: Gebruiker
"""

#Libraries
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter
from datetime import datetime
import matplotlib.dates as mdates
import math

import statsmodels.formula.api as smf

pd.options.mode.chained_assignment = None  # default='warn'
sys.path.append('C:/Users/Gebruiker/Desktop/Bayesian Statistics/covid19-main')

import load_data

data = load_data.df
data = data.dropna()

data=data.groupby('Date').agg({'RNA_Flow':'sum',
                               'Total_reported':'sum'})

url = "https://raw.githubusercontent.com/mzelst/covid-19/master/data/all_data.csv"
NICE_admission = pd.read_csv(url, sep = ",")
NICE_admission.index = NICE_admission['date']
NICE_admission = NICE_admission['Hospital_Intake_Proven']
NICE_admission.index = pd.to_datetime(NICE_admission.index)
before_end_date = '2021-05-23'
NICE_admission = NICE_admission[NICE_admission.index <= before_end_date]
after_start_date = '2020-09-07'
NICE_admission = NICE_admission[ NICE_admission.index>= after_start_date]
NICE_admission = NICE_admission.rename("Hospital_admission")

data = data[ data.index <= datetime.strptime(before_end_date, '%Y-%m-%d').date() ]
data = data[ data.index >= datetime.strptime(after_start_date, '%Y-%m-%d').date() ]
data['Hospital_admission'] = NICE_admission

data['RNA_Flow'] = np.log(data['RNA_Flow'])
data.loc[data['RNA_Flow'] == - math.inf, ['RNA_Flow']] = 29

X = data[['RNA_Flow','Total_reported']]
y = data[['Hospital_admission']]

data['Date'] = pd.to_datetime(data.index)
data['dayOfWeek'] = data['Date'].dt.day_name()


dayData = data.groupby('dayOfWeek').mean()
dayData['dayOfWeek'] = dayData.index

weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
mapping = {day: i for i, day in enumerate(weekdays)}
key = dayData['dayOfWeek'].map(mapping)
dayData = dayData.iloc[key.argsort()]


data = data.reset_index(drop = True)

#ARDL 

def createLag(data, variable, n):
    
    data.sort_values(['Date'])
    
    lagName = f"L{n}_{variable}"
    data[lagName] = np.nan
    
    for index in range(n+1,len(data[variable])):
        data[lagName][index] = data[variable][index - n] 
    return data

data = createLag(data, "RNA_Flow", 7)
data = createLag(data, "RNA_Flow", 14)
data = createLag(data, "Total_reported", 7)
data = createLag(data, "Total_reported", 14)
data = createLag(data, "Hospital_admission", 7)
data = createLag(data, "Hospital_admission", 14)

end_train_data = datetime.strptime("2021-03-06" , '%Y-%m-%d')
df = data[ data['Date']<= end_train_data]

ARDL_7_est = smf.ols(formula = 'Hospital_admission ~ 1 + L7_Hospital_admission + L7_RNA_Flow + L7_Total_reported', data = df)
ARDL_7_fit = ARDL_7_est.fit()
ARDL_7_fit.params
print(ARDL_7_fit.summary())

ARDL_7_14_est = smf.ols(formula = 'Hospital_admission ~ 1 + L7_Hospital_admission  + L14_Hospital_admission + L7_RNA_Flow + L14_RNA_Flow + L7_Total_reported + L14_Total_reported', data = df)
ARDL_7_14_fit = ARDL_7_14_est.fit()
ARDL_7_14_fit.params
print(ARDL_7_14_fit.summary())

data['Predict_L7'] = ""
data['Predict_L7'] =  ARDL_7_fit.params[0] + data["L7_RNA_Flow"] * ARDL_7_fit.params[1] + data["L7_Total_reported"] * ARDL_7_fit.params[2]

data['Predict_L7_L14'] = ""
data['Predict_L7_L14'] = ARDL_7_14_fit.params[0] + data["L7_RNA_Flow"] * ARDL_7_14_fit.params[1] + data["L14_RNA_Flow"] * ARDL_7_14_fit.params[2] + data["L7_Total_reported"] * ARDL_7_14_fit.params[3] + data["L14_Total_reported"] * ARDL_7_14_fit.params[4]

#Add day-dummies
data  = data.join(pd.get_dummies(data['dayOfWeek']) )
df = data[ data['Date']<= end_train_data]

ARDL_7_est = smf.ols(formula = 'Hospital_admission ~ 1  + L7_Hospital_admission + L7_RNA_Flow + L7_Total_reported + Monday + Tuesday + Wednesday + Thursday + Friday + Saturday', data = df)
ARDL_7_fit = ARDL_7_est.fit()
ARDL_7_fit.params
print(ARDL_7_fit.summary())

ARDL_7_14_est = smf.ols(formula = 'Hospital_admission ~ 1 + L7_Hospital_admission  + L14_Hospital_admission  + L7_RNA_Flow + L14_RNA_Flow + L7_Total_reported + L14_Total_reported  + Monday + Tuesday + Wednesday + Thursday + Friday + Saturday', data = df)
ARDL_7_14_fit = ARDL_7_14_est.fit()
ARDL_7_14_fit.params
print(ARDL_7_14_fit.summary())

data['Predict_L7'] = ""
data['Predict_L7'] =  ARDL_7_fit.params.Intercept + data["L7_Hospital_admission"] * ARDL_7_fit.params.L7_Hospital_admission + data["L7_RNA_Flow"] * ARDL_7_fit.params.L7_RNA_Flow + data["L7_Total_reported"] * ARDL_7_fit.params.L7_Total_reported + data["Monday"] * ARDL_7_fit.params.Monday +  data["Tuesday"] * ARDL_7_fit.params.Tuesday +  data["Wednesday"] * ARDL_7_fit.params.Wednesday +  data["Thursday"] * ARDL_7_fit.params.Thursday +  data["Friday"] * ARDL_7_fit.params.Friday +  data["Saturday"] * ARDL_7_fit.params.Saturday 

data['Predict_L7_L14'] = ""
data['Predict_L7_L14'] = ARDL_7_14_fit.params.Intercept + data["L7_Hospital_admission"] * ARDL_7_14_fit.params.L7_Hospital_admission + data["L14_Hospital_admission"] * ARDL_7_14_fit.params.L14_Hospital_admission + data["L7_RNA_Flow"] * ARDL_7_14_fit.params.L7_RNA_Flow + data["L14_RNA_Flow"] * ARDL_7_14_fit.params.L14_RNA_Flow + data["L7_Total_reported"] * ARDL_7_14_fit.params.L7_Total_reported  + data["L14_Total_reported"] * ARDL_7_14_fit.params.L14_Total_reported + data["Monday"] * ARDL_7_14_fit.params.Monday +  data["Tuesday"] * ARDL_7_14_fit.params.Tuesday +  data["Wednesday"] * ARDL_7_14_fit.params.Wednesday +  data["Thursday"] * ARDL_7_14_fit.params.Thursday +  data["Friday"] * ARDL_7_14_fit.params.Friday +  data["Saturday"] * ARDL_7_14_fit.params.Saturday

totalData = data

data = data[data["Date"]> end_train_data]

dataPredict = pd.DataFrame(data["Date"], columns = ["Date"])

dataPredict["y"]= data["Hospital_admission"]
dataPredict["yhat_L7"] = data["Predict_L7"]
dataPredict["yhat_L14"] = data["Predict_L7_L14"]
    
n = len(dataPredict["y"]) 
y = dataPredict["y"].to_numpy().reshape(1,n)
yhat_L7 = dataPredict["yhat_L7"].to_numpy().reshape(1,n)
yhat_L14 = dataPredict["yhat_L14"].to_numpy().reshape(1,n)

MAE_ARDL_L7 = 1 / n  *  abs( y - yhat_L7 ).sum()
MAPE_ARDL_L7 = 100 / n * (abs(y[y!=0] - yhat_L7[y!=0])/y[y!=0]).sum()
RMSE_ARDL_L7 = math.sqrt( 1 / n * ((y - yhat_L7) ** 2).sum() )
MAE_ARDL_L7, MAPE_ARDL_L7, RMSE_ARDL_L7

MAE_ARDL_L14 = 1 / n  *  abs( y - yhat_L14 ).sum()
MAPE_ARDL_L14 = 100 / n * (abs(y[y!=0] - yhat_L14[y!=0])/y[y!=0]).sum()
RMSE_ARDL_L14 = math.sqrt( 1 / n * ((y - yhat_L14) ** 2).sum() )
MAE_ARDL_L14, MAPE_ARDL_L14, RMSE_ARDL_L14

data = totalData