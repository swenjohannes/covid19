# -*- coding: utf-8 -*-
"""
Created on Fri May  7 11:28:31 2021

@author: Swen, Rob, Peter
"""

#Libraries
import pandas as pd

# Import all datasets
import load_hospital
import load_sewage
import load_los

#get data
dfHos = load_hospital.dfHos
dfS = load_sewage.dfRegion
dfLoS =  load_los.dfLoS

# Merge hospital & sewage data 
df = pd.merge(dfS, dfHos, on = ['Date'], how = 'left')

df

# end date of the data used is 2021-05-23
df.index = df['Date']
df=df.drop(columns = 'Date')
df.index = pd.to_datetime(df.index)
before_end_date = df.index<='2021-05-23'
df=df[before_end_date]
