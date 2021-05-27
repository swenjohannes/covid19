# -*- coding: utf-8 -*-
"""
Created on Fri May  7 11:28:26 2021

@author: Swen, Peter, Rob
"""

#Import Required Libraries 
import pandas as pd
from datetime import date, timedelta

#Import Sewage RNA Data
url = "https://data.rivm.nl/covid-19/COVID-19_rioolwaterdata.csv"
dfS = pd.read_csv(url, sep = ";")
dfS.rename(columns = {'Date_measurement':'Date', 
                      'RWZI_AWZI_name':'location', 
                      'RWZI_AWZI_code':'Code',
                      'RNA_flow_per_100000':'RNA_flow'}, inplace = True)
dfS.drop(columns = 'Version', inplace = True)
dfS['Date'] = pd.to_datetime(dfS['Date'], format = '%Y-%m-%d').dt.date

def code_to_region(code):
        if code < 2000 or code == 33001:
            return "Gr"
        elif code < 3000:
            return "Fr"
        elif code < 4000:
            return "Dr"
        elif code < 6000:
            return "Ov"
        elif code < 7000:
            return "Fl"
        elif code < 10000 or code == 16009:
            return "Ge"
        elif code < 11000 or (code > 14000 and code < 15000):
            return "Ut"
        elif code < 14000 or code == 31005:
            return "Nh"
        elif code < 16000 or (code >= 16010 and code <18000):
            return "Zh"
        elif code < 25000:
            return "Ze"
        elif code < 30000 or code == 32002:
           return "Nb"
        elif code < 31000:
            return "Lm"
        else:
            raise TypeError("Area Code Not Covered")    
dfS['Region'] = dfS['Code'].apply(code_to_region)


timeframe = pd.date_range(start = "2020-09-07", end = date.today()).tolist()
dfRegion = pd.DataFrame(timeframe , columns = ["Date"])
dfRegion['Date'] = dfRegion['Date'].dt.date

def RNA_Measure(data, measureDate, selectedRegion, inhabitants_region, surroundingDays = 10):
    
    
    measureValues = dfS[(dfS['Date'] >= measureDate - timedelta(days = surroundingDays)) & 
                       (dfS['Date'] <= measureDate + timedelta(days = surroundingDays)) & 
                       (dfS['Region'] == selectedRegion)]
    
    measurement = measureValues['RNA_flow'].mean()
    
    measurement = measurement*inhabitants_region
    
    return measurement

regions = ['Dr', 'Fl', 'Fr', 'Ge', 'Gr', 'Lm', 'Nb', 'Nh', 'Ov', 'Ut', 'Ze', 'Zh']
inhabitants_per_region = [495, 428, 651, 2097, 587, 1116, 2574, 2888, 1166, 1361, 385, 3726]

dfRegion

for i in range(0,len(regions)):
    region = regions[i]
    inhabitants = inhabitants_per_region[i]
    dfRegion[region] = ""
    for index in range(0,len(dfRegion)):
        dfRegion[region][index] = RNA_Measure(dfS, dfRegion['Date'][index], region, inhabitants)

dfRegion = dfRegion.melt(id_vars=['Date'], value_vars = regions,
                         var_name='Province', value_name='RNA_Flow')

dfRegion['RNA_Flow'] =  pd.to_numeric(dfRegion['RNA_Flow'])
