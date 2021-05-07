# -*- coding: utf-8 -*-
"""
Created on Fri May  7 11:28:31 2021

@author: Swen
"""

#Libraries
import pandas as pd

# Import all datasets
import load_hospital
import load_sewage
import load_los

#get data
dfHos = load_hospital.dfHos
dfRegion = load_sewage.dfRegion
dfLoS =  load_los.dfLoS

# Merge hospital & sewage data 
df = pd.merge(dfRegion, dfHos, on = ['Date', 'Province'], how = 'left')

