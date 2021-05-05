# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:22:43 2021

@author: peter
"""

import pandas as pd
import os
os.chdir('./covid19')

#download & save data local
current_data = pd.read_csv('https://data.rivm.nl/covid-19/COVID-19_aantallen_gemeente_per_dag.csv',sep=';')



# Put cases reported in a time series per province
reported_cases_per_province = current_data['Date_of_publication'].unique()
reported_cases_per_province = pd.DataFrame(reported_cases_per_province)
reported_cases_per_province.columns = ['Date']
for provinces in current_data['Province'].unique():
    is_province = current_data['Province']== provinces
    province_data = current_data[is_province]
    
    place = 0
    reported_in_province = []
    for date in province_data['Date_of_publication'].unique():
        place = place +1
        is_date = province_data['Date_of_publication']==date
        reported_in_province.append(province_data[is_date]['Total_reported'].sum())
    reported_cases_per_province[provinces]=(reported_in_province)
    
    
# Put hospital admissions in a time series per province
hospital_admissions_per_province = current_data['Date_of_publication'].unique()
hospital_admissions_per_province = pd.DataFrame(hospital_admissions_per_province)
hospital_admissions_per_province.columns = ['Date']
for provinces in current_data['Province'].unique():
    is_province = current_data['Province']== provinces
    province_data = current_data[is_province]
    
    place = 0
    hospital_in_province = []
    for date in province_data['Date_of_publication'].unique():
        place = place +1
        is_date = province_data['Date_of_publication']==date
        hospital_in_province.append(province_data[is_date]['Hospital_admission'].sum())
    hospital_admissions_per_province[provinces]=(hospital_in_province)
    

        