# -*- coding: utf-8 -*-
"""
Created on Fri May  7 12:21:17 2021

@author: peter
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
from functions import *

import load_data
data = load_data.df

#Plot hospital data
plt.plot(data.Hospital_admission)

#Add dummies for day of the week, drop sunday
dummies = pd.get_dummies(pd.to_datetime(data.index).day_name()) \
            .drop(columns = 'Sunday')

data = pd.concat([data.reset_index(), dummies], axis = 1) \
            .set_index('Date')

# Adding the lag of the possitive tests and RNA from from 1 to 3 weeks back
for i in range(7,22):
    data["Total_reported_lag_{}".format(i)] = data.Total_reported.shift(i)
    data["RNA_Flow_lag_{}".format(i)] = data.RNA_Flow.shift(i)
    
data=data.dropna()  
data = data.drop(columns=['RNA_Flow', 'Total_reported'])

#Standardize variables

# the y is the dependent variable, IC, practice on hospital admissions
y = data.Hospital_admission
X = data.drop(['Hospital_admission'], axis=1)

X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y)

# rmove first 7 entries test data
X_test = X_test.iloc[7:]
y_test = y_test.iloc[7:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

coef_names = X_train.columns 

#Fit a linear regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

plotModelResults(lr, X_train_scaled, X_test_scaled, y_train, y_test, plot_intervals=True)
plotCoefficients(lr, coef_names)

#Fit Lasso regression
lambdas = list(np.arange(0.001,2,0.001))
lasso = LassoCV(max_iter=10000, cv=tscv,alphas=lambdas)
lasso.fit(X_train_scaled, y_train)

plotModelResults(lasso, X_train_scaled, X_test_scaled, y_train, y_test, plot_intervals=True)
plotCoefficients(lasso, coef_names) 

#Fit Ridge regression
ridge = RidgeCV(cv=tscv,alphas=lambdas)
ridge.fit(X_train_scaled, y_train)

plotModelResults(ridge, X_train_scaled, X_test_scaled, y_train, y_test, plot_intervals=True)
plotCoefficients(ridge, coef_names)

#Extract coefficients and write to csv
coefficients_lasso = pd.Series(lasso.coef_,index=X_train.columns.values)
coefficients_lasso.to_csv('lasso_coef.csv',header=False)    
coefficients_ridge = pd.Series(ridge.coef_,index=X_train.columns.values)
coefficients_ridge.to_csv('ridge_coef.csv',header=False)
coefficients_lr = pd.Series(lr.coef_,index=X_train.columns.values)
coefficients_lr.to_csv('lr_coef.csv',header=False)

#Plot train periods -> use train as test!
plotModelResults(lr, X_train_scaled, X_train_scaled, y_train, y_train)
plotModelResults(lasso, X_train_scaled, X_train_scaled, y_train, y_train )
plotModelResults(ridge, X_train_scaled, X_train_scaled, y_train, y_train )
