# -*- coding: utf-8 -*-

#Libraries
import sys
import numpy as np
import pandas as pd
from datetime import timedelta


sys.path.append('C:/Users/Gebruiker/Desktop/Bayesian Statistics/covid19-main')


import Bs_ARDL


def predict(modelParam, data, daysAhead):

    #Day-Dummy Value
    day = (data['Date'].iloc[-1] + timedelta(days = daysAhead)).weekday()
    if day == 6:
        dayValue = 0
    else:
        dayValue = modelParam['Coefficient'][day]
    
    trCoeff = modelParam['Coefficient'][6:21].to_numpy()
    trLagValues = data['Total_reported'][0 + (daysAhead - 1):15 + (daysAhead - 1)].to_numpy()
    trValue = np.matmul( trCoeff,np.flip(trLagValues))
    
    RNACoeff = modelParam['Coefficient'][21:36].to_numpy()
    RNALagValues = data['RNA_Flow'][0 + (daysAhead - 1):15 + (daysAhead - 1)].to_numpy()
    RNAvalue = np.matmul(RNACoeff, np.flip(RNALagValues))
    
    return(dayValue + trValue + RNAvalue)


def createForecast():
    
    data = Bs_ARDL.data
    
    #select data from last 21 periods
    data = data.iloc[-21:,:]
    data = data.reset_index(drop = True)
    
    lassoParam = pd.read_csv("lasso_coef.csv", names = ["Variable","Coefficient"])
    lrParam = pd.read_csv("lr_coef.csv", names = ["Variable","Coefficient"])
    ridgeParam = pd.read_csv("ridge_coef.csv", names = ["Variable","Coefficient"])
    
    models = [["Lasso", lassoParam], ["Linear", lrParam], ["Ridge", ridgeParam]]
    
    #Create forecasting Value Framework
    forecast = { 'Date': pd.date_range(start = data['Date'].iloc[-1] + timedelta(days = 1), periods = 7, freq = 'D') , 'DaysAhead': range(1,8)}
    forecast = pd.DataFrame(forecast)
    
    #Forecast for the different ML models
    for model in models:
        values = [0] * len(forecast['DaysAhead'])
        for n in range(0, len(forecast['DaysAhead'])):
            values[n] = predict(model[1], data, n + 1)
            
        forecast[f"{model[0]}"] = values
    

    #Lastly forecast for ARDL models    
    ARDL_7 = Bs_ARDL.ARDL_7_fit.params
    ARDL_14 = Bs_ARDL.ARDL_7_14_fit.params
    
    forecast["ARDL_7"]  = ""
    forecast["ARDL_7_14"] = ""
        
    for n in range(0 , len(forecast["DaysAhead"])):
        day = (data['Date'].iloc[-1] + timedelta(days = n + 1)).weekday()
        if day == 6:
            dayValueL7 = dayValueL14 = 0
        else:
            dayValueL7 = ARDL_7[3 + day]
            dayValueL14 = ARDL_14[3 + day]
        
        L7_RNA = data["RNA_Flow"][20 - 7 + n]
        L14_RNA = data["RNA_Flow"][20 - 14 + n]
        
        L7_Rep  = data["Total_reported"][20 - 7 + n]
        L14_Rep = data["Total_reported"][20 - 14 + n]
        
        L7_Hos  = data["Hospital_admission"][20 - 7 + n]
        L14_Hos = data["Hospital_admission"][20 - 14 + n]
        
        forecast["ARDL_7"][n] = ARDL_7.Intercept + L7_Hos * ARDL_7.L7_Hospital_admission + dayValueL7 +  ARDL_7.L7_RNA_Flow * L7_RNA + ARDL_7.L7_Total_reported * L7_Rep
        forecast["ARDL_7_14"][n] = ARDL_14.Intercept + L7_Hos * ARDL_14.L7_Hospital_admission + L14_Hos * ARDL_14.L14_Hospital_admission + dayValueL14 +  ARDL_14.L7_RNA_Flow * L7_RNA +  ARDL_14.L14_RNA_Flow * L14_RNA + ARDL_14.L7_Total_reported * L7_Rep + ARDL_14.L14_Total_reported * L14_Rep
    
    return forecast

data = Bs_ARDL.data