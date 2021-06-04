# -*- coding: utf-8 -*-
"""
Created on Fri May  7 11:28:26 2021
@author: Swen, Peter, Rob
"""

#Import Required Libraries 
import pandas as pd

#Import hospital data
url = 'https://raw.githubusercontent.com/mzelst/covid-19/master/data/all_data.csv'
dfHos = pd.read_csv(url, sep=',')
dfHos = dfHos[['date', 'positivetests', 'Hospital_Intake_Proven']]
dfHos.rename(columns = {'date':'Date',
                        'positivetests':'Total_reported',
                        'Hospital_Intake_Proven': 'Hospital_admission'},
             inplace = True)

dfHos.Date = pd.to_datetime(dfHos.Date, format = '%Y-%m-%d').dt.date
