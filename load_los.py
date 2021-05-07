# -*- coding: utf-8 -*-
"""
Created on Fri May  7 11:28:27 2021

@author: Swen, Peter, Rob
"""
#Libraries
import pandas as pd
from datetime import date, timedelta

#Import LoS data
url = "https://raw.githubusercontent.com/mzelst/covid-19/master/data-nice/treatment-time/IC/nice_daily_treatment-time_IC_"
#read yesterdays data! (current might not be available yet)
dfLoS = pd.read_csv(url + str(date.today() - timedelta(1)) + ".csv") 

dfLoS.rename(columns = {'dagen':'Dagen',
                           'IC_to_clinical':'hospitalized',
                           'Treatment_time_hospitalized':'current',
                           'Treatment_time_to_exit':'exit',
                           'Treatment_time_to_death':'death'}, inplace = True)

dfLoS.set_index('Dagen')