# -*- coding: utf-8 -*-
"""
Created on Fri May  7 12:21:17 2021

@author: peter
"""

import load_data

data, dataLoS = load_data.df, load_data.dfLoS


data=data.groupby('Date').agg({'RNA_Flow':'sum',
                               'Hospital_admission':'sum',
                               'Total_reported':'sum'})
                  

import matplotlib.pyplot as plt 

plt.plot(data.Hospital_admission)


import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Adding the lag of the target variable from 1 steps back up to 7. So of one week back
#for i in range(1, 7):
  #  data["lag_{}".format(i)] = data.Hospital_admission.shift(i)
  
# Maybe add the one week average of hospital admissions over the last week?



    
# Adding the lag of the possitive tests from from 1 to 3 weeks back
for i in [7,14,21]:
    data["Total_reported_lag_{}".format(i)] = data.Total_reported.shift(i)
    
# Adding the lag of the possitive tests from 1 to 3 weeks back
for i in [7,14,21]:
    data["RNA_Flow_lag_{}".format(i)] = data.RNA_Flow.shift(i)
    

    

    

# for time-series cross-validation set 5 folds 
tscv = TimeSeriesSplit(n_splits=5)

# Define a function for splitting the data into train and test data
def timeseries_train_test_split(X, y, test_size):
    """
        Perform train-test split with respect to time series structure
    """
    
    # get the index after which test set starts
    test_index = int(len(X)*(1-test_size))
    
    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]
    
    return X_train, X_test, y_train, y_test

# the y is the dependent variable, IC, practice on hospital admissions
y = data.dropna().Hospital_admission
X = data.dropna().drop(['Hospital_admission'], axis=1)


X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)

#perform a linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def plotModelResults(model, X_train=X_train, X_test=X_test, plot_intervals=False, plot_anomalies=False):
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
        
        if plot_anomalies:
            anomalies = np.array([np.NaN]*len(y_test))
            anomalies[y_test<lower] = y_test[y_test<lower]
            anomalies[y_test>upper] = y_test[y_test>upper]
            plt.plot(anomalies, "o", markersize=10, label = "Anomalies")
    
    error = mean_absolute_percentage_error(prediction, y_test)
    plt.title("Mean absolute percentage error {0:.2f}%".format(error))
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
    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)
    
    plt.figure(figsize=(15, 7))
    coefs.coef.plot(kind='bar')
    plt.grid(True, axis='y')
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed');
    
plotModelResults(lr, plot_intervals=True)
plotCoefficients(lr)





from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

plotModelResults(lr, X_train=X_train_scaled, X_test=X_test_scaled, plot_intervals=True)
plotCoefficients(lr)


from sklearn.linear_model import LassoCV, RidgeCV

ridge = RidgeCV(cv=tscv)
ridge.fit(X_train_scaled, y_train)

plotModelResults(ridge, 
                 X_train=X_train_scaled, 
                 X_test=X_test_scaled, 
                 plot_intervals=True, plot_anomalies=True)
plotCoefficients(ridge)


lasso = LassoCV(max_iter=10000, cv=tscv)
lasso.fit(X_train_scaled, y_train)

plotModelResults(lasso,
                 X_train=X_train_scaled, 
                 X_test=X_test_scaled, 
                 plot_intervals=True, plot_anomalies=True)
plotCoefficients(lasso)


# to see how it fits on train data, change rule 95 to y.train instead of y.test
plotModelResults(lasso,
                 X_train=X_train_scaled, 
                 X_test=X_train_scaled, 
                 plot_intervals=False, plot_anomalies=True)


plotModelResults(ridge, 
                 X_train=X_train_scaled, 
                 X_test=X_train_scaled, 
                 plot_intervals=False, plot_anomalies=True)
plotCoefficients(ridge)



#calc mean absolute percentage error using the previous number
y_predic=y_test.shift(1)
y_test.drop(y_test.tail(1).index,inplace=True)
y_test.drop(y_test.head(1).index,inplace=True)

y_predic.drop(y_predic.tail(1).index,inplace=True)
y_predic.drop(y_predic.head(1).index,inplace=True)

mean_absolute_percentage_error(y_test,y_predic)


