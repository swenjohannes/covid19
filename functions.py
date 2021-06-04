# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:54:45 2021

@author: Swen
"""
import pandas as pd
import numpy as np
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