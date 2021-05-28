# -*- coding: utf-8 -*-
"""
Created on Fri May  7 12:21:17 2021

@author: peter
"""
import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import math

import load_data

data, dataLoS = load_data.df, load_data.dfLoS


data=data.groupby('Date').agg({'RNA_Flow':'sum',
                               'Hospital_admission':'sum',
                               'Total_reported':'sum'})


data.index = pd.to_datetime(data.index)
before_end_date = data.index<='2021-05-23'
data=data[before_end_date]

#log of RNA flow already done before importing data? Else
data['RNA_Flow'] = np.log(data['RNA_Flow'])

# add dummies for days of the week
week_days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
for i in range(0,6):
    data[week_days[i]] = (data.index.weekday == i).astype(int)

import matplotlib.pyplot as plt 

plt.plot(data.Hospital_admission)

    
# Adding the lag of the possitive tests from from 1 to 3 weeks back
for i in range(7,22):
    data["Total_reported_lag_{}".format(i)] = data.Total_reported.shift(i)
    
# Adding the lag of the possitive tests from 1 to 3 weeks back
for i in range(7,22):
    data["RNA_Flow_lag_{}".format(i)] = data.RNA_Flow.shift(i)
    

data=data.dropna() 
data = data.drop(columns=['RNA_Flow'])
data = data.drop(columns=['Total_reported'])

    

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

# the y is the dependent variable, IC, practice on hospital admissions
y = data.Hospital_admission
X = data.drop(['Hospital_admission'], axis=1)


X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y)

# rmove first 7 entries test data
X_test = X_test.iloc[7:]
y_test = y_test.iloc[7:]


def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred)))

                   
def root_mean_square_deviation(y_true, y_pred):
    return math.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))

def plotModelResults(model, X_train=X_train, X_test=X_test, plot_intervals=False):
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
        
       
    MAPE = mean_absolute_percentage_error(prediction, y_test)
    MAE = mean_absolute_error(prediction, y_test)
    RMSE = root_mean_square_deviation(prediction, y_test)
    
    MAPE = round(MAPE, 2)
    MAE = round(MAE, 2)
    RMSE = round(RMSE, 2)
    
    plt.title("MAPE: {}%, MAE: {}, RSE: {}".format(MAPE, MAE, RMSE))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True);
    
def plotCoefficients(model):
    """
        Plots sorted coefficient values of the model
    """
    
    coefs = pd.DataFrame(model.coef_, X_train.columns)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    #coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)
    
    plt.figure(figsize=(15, 7))
    coefs.coef.plot(kind='bar')
    plt.grid(True, axis='y')
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed');
    





from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

plotModelResults(lr, X_train=X_train_scaled, X_test=X_test_scaled, plot_intervals=True)
plotCoefficients(lr)


from sklearn.linear_model import LassoCV, RidgeCV

lambdas = list(np.arange(0,20,0.1))

lasso = LassoCV(max_iter=10000, cv=tscv,alphas=lambdas)
lasso.fit(X_train_scaled, y_train)


ridge = RidgeCV(cv=tscv,alphas=lambdas)
ridge.fit(X_train_scaled, y_train)

plotModelResults(ridge, 
                 X_train=X_train_scaled, 
                 X_test=X_test_scaled, 
                 plot_intervals=True)
plotCoefficients(ridge)



plotModelResults(lasso,
                 X_train=X_train_scaled, 
                 X_test=X_test_scaled, 
                 plot_intervals=True)
plotCoefficients(lasso)


coefficients_lasso = pd.Series(lasso.coef_,index=X_train.columns.values)
coefficients_lasso.to_csv('lasso_coef.csv',header=False)    
coefficients_ridge = pd.Series(ridge.coef_,index=X_train.columns.values)
coefficients_ridge.to_csv('ridge_coef.csv',header=False)
coefficients_lr = pd.Series(lr.coef_,index=X_train.columns.values)
coefficients_lr.to_csv('lr_coef.csv',header=False)

# to see how it fits on train data, change rule 95 to y.train instead of y.test


def plotModelFit(model, X_train=X_train, X_test=X_test, plot_intervals=False):
    """
        Plots modelled vs fact values, prediction intervals and anomalies
    
    """
    
    prediction = model.predict(X_test)
    
    plt.figure(figsize=(15, 7))
    plt.plot(prediction, "g", label="prediction", linewidth=2.0)
    plt.plot(y_train.values, label="actual", linewidth=2.0)
    
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
    
    
    MAPE = mean_absolute_percentage_error(prediction, y_train)
    MAE = mean_absolute_error(prediction, y_train)
    RMSE = root_mean_square_deviation(prediction, y_train)
    
    MAPE = round(MAPE, 2)
    MAE = round(MAE, 2)
    RMSE = round(RMSE, 2)
    
    plt.title("MAPE: {}%, MAE: {}, RSE: {}".format(MAPE, MAE, RMSE))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True);

plotModelFit(lr,
                 X_train=X_train_scaled, 
                 X_test=X_train_scaled, 
                 plot_intervals=False)

plotModelFit(lasso,
                 X_train=X_train_scaled, 
                 X_test=X_train_scaled, 
                 plot_intervals=False)


plotModelFit(ridge, 
                 X_train=X_train_scaled, 
                 X_test=X_train_scaled, 
                 plot_intervals=False)




