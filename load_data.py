# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:22:43 2021

@author: peter, swen, rob
"""

#Import Required Libraries & functions
import pandas as pd
from datetime import date, timedelta

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
            
def RNA_Measure(data, measureDate, selectedRegion, surroundingDays = 10):
    
    
    measureValues = dfS[(dfS['Date'] >= measureDate - timedelta(days = surroundingDays)) & 
                       (dfS['Date'] <= measureDate + timedelta(days = surroundingDays)) & 
                       (dfS['Region'] == selectedRegion)]
    
    measurement = measureValues['RNA_flow'].mean()
    
    return measurement

#Import hospital data
url = 'https://data.rivm.nl/covid-19/COVID-19_aantallen_gemeente_per_dag.csv'
dfHos = pd.read_csv(url, sep=';')
dfHos.rename(columns = {'Date_of_publication' : 'Date'}, inplace = True)
dfHos['Date'] = pd.to_datetime(dfHos['Date'], format = '%Y-%m-%d').dt.date


#group by date and province & aggregate
dfHos = dfHos.groupby(['Date', 'Province']).agg({'Hospital_admission': 'sum', 
                                          'Total_reported' : 'sum'}).reset_index()

regions = {'Drenthe' : 'Dr',
          'Flevoland' : 'Fl', 
          'Fryslân' : 'Fr', 
          'Gelderland' : 'Ge',
          'Groningen' : 'Gr',
          'Limburg' : 'Lm', 
          'Noord-Brabant' : 'Nb', 
          'Noord-Holland' : 'Nh',
          'Overijssel' : 'Ov',
          'Utrecht' : 'Ut', 
          'Zeeland' : 'Ze', 
          'Zuid-Holland' : 'Zh'}

dfHos.replace(regions, inplace = True)

#Import Sewage RNA Data
url = "https://data.rivm.nl/covid-19/COVID-19_rioolwaterdata.csv"
dfS = pd.read_csv(url, sep = ";")
dfS.rename(columns = {'Date_measurement':'Date', 
                      'RWZI_AWZI_name':'location', 
                      'RWZI_AWZI_code':'Code',
                      'RNA_flow_per_100000':'RNA_flow'}, inplace = True)
dfS.drop(columns = 'Version', inplace = True)
dfS['Date'] = pd.to_datetime(dfS['Date'], format = '%Y-%m-%d').dt.date
dfS['Region'] = dfS['Code'].apply(code_to_region)


timeframe = pd.date_range(start = "2020-09-07", end = date.today()).tolist()
dfRegion = pd.DataFrame(timeframe , columns = ["Date"])
dfRegion['Date'] = dfRegion['Date'].dt.date



for region in regions.values():
    dfRegion[region] = ""
    for index in range(0,len(dfRegion)):
        dfRegion[region][index] = RNA_Measure(dfS, dfRegion['Date'][index], region)

dfRegion = dfRegion.melt(id_vars=['Date'], value_vars = regions.values(),
                         var_name='Province', value_name='RNA_Flow')

dfRegion['RNA_Flow'] =  pd.to_numeric(dfRegion['RNA_Flow'])

#merge the files together
data = pd.merge(dfRegion, dfHos, on = ['Date', 'Province'], how = 'left') 

#Import LoS data!
url = "https://raw.githubusercontent.com/mzelst/covid-19/master/data-nice/treatment-time/IC/nice_daily_treatment-time_IC_"
#read yesterdays data! (current might not be available yet)
dataLoS = pd.read_csv(url + str(date.today() - timedelta(1)) + ".csv") 

#Final result
data
dataLoS
