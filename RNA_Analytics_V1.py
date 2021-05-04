# -*- coding: utf-8 -*-
"""
Created on Mon May  3 09:28:52 2021

@author: 
"""

#Import Required Libraries

import requests
import pandas as pd
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta


#-------------------------



#Import Hospitilization Data

url = "https://raw.githubusercontent.com/mzelst/covid-19/master/data/all_data.csv"

dfH = pd.read_csv(url)
dfH['date'] = pd.to_datetime(dfH['date'], format = '%Y-%m-%d')
response = requests.get(url)

#-------------------------


#Import Sewage RNA Data

url = "https://data.rivm.nl/covid-19/COVID-19_rioolwaterdata.csv"

dfS = pd.read_csv(url, sep = ";")
dfS = dfS.rename(columns = {'Date_measurement':'date', 
                            'RWZI_AWZI_name':'location', 
                            'RWZI_AWZI_code':'code',
                            'RNA_flow_per_100000':'RNA_flow'})

dfS['date'] = pd.to_datetime(dfS['date'], format = '%Y-%m-%d').dt.date
locations = dfS.location.value_counts()


def set_Region(df):
    
    df['Region'] = ""

    for index in range(0,len(df['date'])):
        
        code = df['code'][index]
        
        if code < 2000 or code == 33001:
            region = "Gr"
        elif code < 3000:
            region = "Fr"
        elif code < 4000:
            region = "Dr"
        elif code < 6000:
            region = "Ov"
        elif code < 7000:
            region = "Fl"
        elif code < 10000 or code == 16009:
            region = "Ge"
        elif code < 11000 or (code > 14000 and code < 15000):
            region = "Ut"
        elif code < 14000 or code == 31005:
            region = "Nh"
        elif code < 16000 or (code >= 16010 and code <18000):
            region = "Zh"
        elif code < 25000:
            region = "Ze"
        elif code < 30000 or code == 32002:
            region = "Nb"
        elif code < 31000:
            region = "Lm"
        else:
            raise TypeError("Area Code Not Covered")
        
        df['Region'][index] = region
    
    return df

def RNA_Measure(dfS, measureDate, selectedRegion, surroundingDays = 10):
    
    
    measureValues = dfS[(dfS['date'] >= measureDate - timedelta(days = surroundingDays)) & 
                       (dfS['date'] <= measureDate + timedelta(days = surroundingDays)) & 
                       (dfS['Region'] == selectedRegion)]
    
    measurement = measureValues['RNA_flow'].mean()
    
    return measurement



regions = ["Gr","Fr","Dr","Ov","Fl","Ge","Ut","Nh","Zh","Ze","Nb","Lm"]
timeframe = pd.date_range(start = "2020-09-07", end = date.today()).tolist()
dfRegion = pd.DataFrame(timeframe , columns = ["date"])
dfRegion['date'] = dfRegion['date'].dt.date

dfS = set_Region(dfS)

for region in regions:
    dfRegion[region] = ""
    for index in range(0,len(dfRegion)):
        dfRegion[region][index] = RNA_Measure(dfS, dfRegion['date'][index], region)

#Plot Basic Data

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(dfH['date'],
        dfH["IC_Current"],
        color='red')
ax.set(xlabel="Date", ylabel="Required IC Capacity",
       title="COVID-19 IC Required Capacity")
plt.xticks(rotation=45)

# Format the x axis
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(DateFormatter("%y-%m-%d"))
plt.show()



fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(dfH['date'],
        dfH["IC_Intake"],
        color='orange')
ax.set(xlabel="Date", ylabel="IC Intake",
       title="COVID-19 IC Daily Intake")
plt.xticks(rotation=45)

# Format the x axis
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(DateFormatter("%y-%m-%d"))
plt.show()





fig, ax = plt.subplots(figsize=(12, 8))
for region in regions:
    ax.plot(dfRegion['date'], dfRegion[region], label = region)
    ax.legend([region])  
      
ax.set(xlabel="Date", ylabel="RNA Flow Value",
       title="RNA Measure Per Province")
ax.legend()
plt.xticks(rotation=45)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(DateFormatter("%y-%m-%d"))

plt.show()