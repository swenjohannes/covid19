# -*- coding: utf-8 -*-
"""
Created on Fri May  7 11:28:26 2021

@author: Swen, Peter, Rob
"""

#Import Required Libraries 
import pandas as pd

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

