# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:44:11 2021

@author: Rob
"""

# Importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as random
from scipy import stats
from scipy.stats import weibull_min


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

# Now we define a weibull function to forecast LoS
def losforecast():
    LoS = np.argmin( kaplanMeier > random.random())
    if(LoS == 0):
        LoS = 1
    #Sample from Kaplan-Meier
    return LoS


# Now we create a patient class
class patient:
    def __init__(self, arrival):
        self.los = losforecast()
        self.arrival = arrival
        self.leave = self.arrival + self.los
        
    def update(self, los):
        self.los = los
        self.leave = self.arrival + self.los
        

#Known arrivals from dataset 

url = "https://raw.githubusercontent.com/mzelst/covid-19/master/data-nice/data-nice-json/2021-05-26.csv"
dfHosp = pd.read_csv(url)
dfHosp['date'] = pd.to_datetime(dfHosp['date'], format = '%Y-%m-%d').dt.date

arrivals = dfHosp["Hospital_Intake_Proven"]


runs = 100



shapeRv = [0]*runs
scaleRv = [0]*runs

for run in range(0,runs):
    #For each time period
    remainingLoS = []
    for period in range(0, len(arrivals)):
        #create num of patients that arrive in that period
        numArrivals = int(arrivals[period])
        for i in range(0,numArrivals):
            y = patient(period)
            y.update(losforecast())
    
            #If they do not leave before the time of observation, note their remaining time
            if y.leave > len(arrivals):
                remainingLoS.append(y.leave - len(arrivals))
    
    data = np.array(remainingLoS)
    shapeRv[run], locR, scaleRv[run] = stats.weibull_min.fit(data, floc=0)

shapeR = np.mean(shapeRv)
scaleR = np.mean(scaleRv)

fig, ax = plt.subplots(figsize=(14, 6))
x = np.linspace(data.min(), 60, 60)
plt.plot(x, weibull_min(shapeR, locR, scaleR).pdf(x),label = "Weibull Remaining LoS", color = "orange")
plt.plot(x, kaplanMeierPDF,label = "Kaplan Meier Total LoS PDF", color = "red")
_ = plt.hist(data, bins=np.linspace(0, 60, 60), density = True, color = "darkgrey")
#plt.title("Weibull fit on Remaining LoS")
plt.xlabel("Remaining Length of Stay")
plt.ylabel("Density")
plt.legend(prop={'size': 8})
plt.savefig(r'C:\Users\Gebruiker\Desktop\graph.png', dpi = 1000)
plt.show()
