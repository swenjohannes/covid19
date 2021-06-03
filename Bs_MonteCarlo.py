# -*- coding: utf-8 -*-
"""
Created on Thu May 27 12:53:06 2021

@author: Gebruiker
"""


import numpy as np
import pandas as pd
import random
from datetime import timedelta
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from datetime import datetime
import matplotlib.pyplot as plt
import sys

sys.path.append('C:/Users/Gebruiker/Desktop/Bayesian Statistics/covid19-main')

import Bs_Forecasts
import Bs_ARDL
import remainingLoS

url = "https://raw.githubusercontent.com/mzelst/covid-19/master/data-nice/data-nice-json/2021-06-02.csv"
dfHosp = pd.read_csv(url)
dfHosp['date'] = pd.to_datetime(dfHosp['date'], format = '%Y-%m-%d').dt.date
before_end_date = datetime.strptime('2021-05-23', "%Y-%m-%d").date()
newDates = dfHosp[dfHosp['date'] > before_end_date][["date", "Hospital_Currently"]]
trueArrivals = dfHosp[dfHosp['date'] > before_end_date][["date", "Hospital_Intake_Proven"]]
dfHosp = dfHosp[dfHosp['date'] <= before_end_date]


data = Bs_ARDL.data 

nationalForecast = { 'Date': pd.date_range(start = data['Date'].iloc[-1] + timedelta(days = 1), periods = 7, freq = 'D') , 'DaysAhead': range(1,8)}
nationalForecast = pd.DataFrame(nationalForecast, columns = ["Date", "Lasso", "Linear", "Ridge", "ARDL_7", "ARDL_7_14"])
nationalForecast = nationalForecast.fillna(0)

forecast = Bs_Forecasts.createForecast()
nationalForecast["Lasso"] = forecast["Lasso"]
nationalForecast["Linear"] = forecast["Linear"]
nationalForecast["Ridge"] = forecast["Ridge"]
nationalForecast["ARDL_7"] = forecast["ARDL_7"]
nationalForecast["ARDL_7_14"] += forecast["ARDL_7_14"]
nationalForecast["LTS"] = [123.47535,  94.60530,  92.04979,  87.28405,  87.34224,  59.16360,  59.14751]
nationalForecast["LLS"] = [129.55645, 105.14008, 107.78151, 107.67362, 114.21071,  81.37030,  84.89123]
nationalForecast["LL"] = [99.42939,  99.45241,  99.97988, 100.59694, 100.55855, 100.06780, 100.24479]
nationalForecast["True"] = trueArrivals["Hospital_Intake_Proven"][0:7].to_numpy()


url = "https://raw.githubusercontent.com/mzelst/covid-19/master/data-nice/treatment-time/Clinical_Beds/nice_daily_treatment-time_clinical_2021-05-26.csv"
dfTreatment = pd.read_csv(url)

d = dfTreatment["Treatment_time_to_exit"] + dfTreatment["Treatment_time_to_death"]
n = dfTreatment["Treatment_time_hospitalized"].sum() + dfTreatment["Treatment_time_to_exit"].sum() + dfTreatment["Treatment_time_to_death"].sum()
kaplanMeier = [0] * 60
kaplanMeier[0] = 1 - (d[0] / n)

for i in range(1, len(kaplanMeier)):
    kaplanMeier[i] = kaplanMeier[i - 1] * (1 - d[i]/n)
    n -= d[i]

kaplanMeier = np.array(kaplanMeier)    

fig, ax = plt.subplots(figsize=(12, 8))
ax.step(range(1,61), kaplanMeier, color = "red") 
ax.set(xlabel="Days Spent On Ward", ylabel="Survival Rate")
fig.show()

kaplanMeierPDF = [0]*len(kaplanMeier)
for i in range(0, len(kaplanMeier)):
    if(i <= len(kaplanMeier) - 2):
        kaplanMeierPDF[i] = kaplanMeier[i] - kaplanMeier[i+1]
    else:
        kaplanMeierPDF[i] = 0

#Create LoS forecast for people currently in the hospital
def LoSforecast(): 
    LoS = np.argmin( kaplanMeier > random.random())
    if(LoS == 0):
        LoS = 60
    #Sample from Kaplan-Meier
    return(LoS)

def rLoSforecast(): 
    rLoS = remainingLoS.scaleR * np.random.weibull(remainingLoS.shapeR,1)
    #Sample Weibull
    return(rLoS)


class patient:
    def __init__(self, arrival, LoS):
        self.los = LoS
        self.arrival = arrival
        self.leave = self.arrival + self.los
        
    def update(self, los):
        self.los = los
        self.leave = self.arrival + self.los
        
def patientExit(day, patientlist):
    for patient in patientlist:
        if(patient.leave <= day + 1):
            patientlist = np.delete(patientlist, np.argwhere(patientlist == patient))
    return patientlist
    
        

# Create for current number of patients in hospital
n = dfHosp["Hospital_Currently"].iloc[-1]

        
def simulation(nruns, forecastValues): 
    
    occupancy = np.zeros((nruns, 7))

    for run in range(0, nruns):
        patientlist = np.array([]) 
        for i in range(0,n):
            patientlist = np.append(patientlist, patient(0, rLoSforecast()))
       
       #Create initial rem LoS
       
       #---

        for day in range(0,7):
            
            newArrivals = int(forecastValues[day])
            
            for newPatient in range(0, newArrivals):
                patientlist = np.append(patientlist, patient(day, LoSforecast()))
            patientlist = patientExit(day, patientlist)
            occupancy[run, day] = len(patientlist)
            
    return occupancy     

simulations = 1000
sim_ARDL_7 = simulation(simulations, nationalForecast["ARDL_7"])
sim_ARDL_14 = simulation(simulations, nationalForecast["ARDL_7_14"])
sim_LTS = simulation(simulations, nationalForecast["LTS"])
sim_LLS = simulation(simulations, nationalForecast["LLS"])
sim_LL = simulation(simulations, nationalForecast["LL"])
sim_True = simulation(simulations, nationalForecast["True"])

def fetchCI(simMatrix, day, percent):
    interval = np.sort(simMatrix[:,day])
    topCI = interval[ int(percent * len(interval))]
    botCI = interval[ int((1-percent) * len(interval))]
    mean = simMatrix[:,day].mean()
    return mean, botCI, topCI


def plotCI(simMatrix, name):
    x = dfHosp["date"][200:454].to_numpy()
    y = dfHosp["Hospital_Currently"][200:454].to_numpy()
    
    xForecast = pd.date_range(start = dfHosp['date'].iloc[-2] + timedelta(days = 1), periods = 8, freq = 'D')
    yForecast = [y[-1]]
    yUpper = [y[-1]]
    yLower = [y[-1]]
    
    for day in range(0,7):
        yForecast = np.append(yForecast, fetchCI(simMatrix, day, 0.95)[0])
        yLower = np.append(yLower, fetchCI(simMatrix, day, 0.95)[1])
        yUpper = np.append(yUpper, fetchCI(simMatrix, day, 0.95)[2])

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x,y, color = "c", label = "Ward Occupancy")
    ax.plot(xForecast,yForecast, color = "red", label = "Forecasted Ward Occupancy")
    ax.plot(newDates['date'], newDates["Hospital_Currently"], color = "indigo", label = "True Occupancy")
    ax.fill_between(xForecast, yUpper, yLower, color = "lightblue", interpolate = True, label = "95% CI")
    ax.set(xlabel="Date", ylabel="Patients in Ward",
           title=f"Ward Demand Forecast Using {name} Model")
    ax.legend()
    plt.xticks(rotation=45)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%y-%m-%d"))
    
    #Zoomed in
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x,y, color = "c", label = "Ward Occupancy")
    ax.plot(xForecast,yForecast, color = "red", label = "Forecasted Ward Occupancy")
    ax.plot(newDates['date'], newDates["Hospital_Currently"], color = "indigo", label = "True Occupancy")
    ax.fill_between(xForecast, yUpper, yLower, color = "lightblue", interpolate = True, label = "95% CI")
    ax.set(xlabel="Date", ylabel="Patients in Ward",
           title=f"Ward Demand Forecast Using {name} Model")
    ax.legend()
    plt.axis([dfHosp['date'].iloc[-2] + timedelta(days = -5), dfHosp['date'].iloc[-2] + timedelta(days = 8), 500, 1500])
    plt.xticks(rotation=45)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%y-%m-%d"))

plotCI(sim_ARDL_7, "ARDL(7)")
plotCI(sim_ARDL_14, "ARDL(14)")
plotCI(sim_LTS, "LTS")
plotCI(sim_LLS, "LLS")
plotCI(sim_LL, "LL")
plotCI(sim_True, "True Arrival")

for i in range(0,7):
    print(fetchCI(sim_LLS, i, 0.95)[1:3])