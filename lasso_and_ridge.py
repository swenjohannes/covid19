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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# for time-series cross-validation set 5 folds 
tscv = TimeSeriesSplit(n_splits=5)

# Define a function for splitting the data into train and test data
def timeseries_train_test_split(X, y):
    """
        Perform train-test split with respect to time series structure
    """
    # get the index after which test set starts
    test_index = 160
    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]
    
    return X_train, X_test, y_train, y_test

# To see how they fit on the train data:
def plotModelResults(model, X_train, X_test, y_train, y_test, plot_intervals=False):
    """
        Plots modelled vs fact values, prediction intervals and anomalies
    """
    
    prediction = model.predict(X_test)
    
    plt.figure(figsize=(15, 7))
    plt.plot(prediction, "g", label="prediction", linewidth=2.0)
    plt.plot(y_test.values, label="actual", linewidth=2.0)
    
    if plot_intervals:
        cv = cross_val_score(model, X_train, y_train, 
                                    cv=tscv, 
                                    scoring="neg_mean_absolute_error")
        mae = cv.mean() * (-1)
        deviation = cv.std()
        
        scale = 1.96
        lower = prediction - (mae + scale * deviation)
        upper = prediction + (mae + scale * deviation)
        
        plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
        plt.plot(upper, "r--", alpha=0.5)
        
       
    MAPE = mean_absolute_percentage_error(y_test, prediction) * 100
    MAE = mean_absolute_error(y_test, prediction)
    RMSE = mean_squared_error(y_test, prediction, squared = False) 
    MSE= mean_squared_error(y_test, prediction, squared = True) 

    plt.title("MAPE: {:.2f}%, MAE: {:.2f}, RMSE: {:.2f}, MSE: {:.2f}" \
              .format(MAPE, MAE, RMSE, MSE))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True);
    
    
def plotCoefficients(model, coef_names):
    
    """
        Plots sorted coefficient values of the model
    """
    coefs = pd.DataFrame(model.coef_, coef_names)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    
    plt.figure(figsize=(15, 7))
    coefs.coef.plot(kind='bar')
    plt.grid(True, axis='y')
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed');
    
#Analysis

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
