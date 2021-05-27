# -*- coding: utf-8 -*-
"""
Created on Sun May 23 10:29:59 2021

@author: peter
"""
import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV

import load_data

data, dataLoS = load_data.df, load_data.dfLoS

# the first 259 are used
start_date=data['Date'].iloc[0]
end_date=data['Date'].iloc[259-1]
all_dates = pd.date_range(start=start_date,end=end_date) 
data['Date'] = pd.to_datetime(data['Date'])
before_end_date = data['Date']<='2021-05-23'
data=data[before_end_date]

#log of RNA flow already done before importing data? Else
data['RNA_Flow'] = np.log(data['RNA_Flow'])

# add dummies for days of the week
week_days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
for i in range(0,6):
    data[week_days[i]] = (data['Date'].dt.weekday == i).astype(int)


#create dataframe which will contain the lags
data_lags = pd.DataFrame()

def add_lags(Province_data):
    #add lags interested in
    # Adding the lag of the possitive tests from 1 up to 3 weeks back
    for i in range(7,22):
        Province_data["Total_reported_lag_{}".format(i)] = Province_data.Total_reported.shift(i)
    # Adding the lag of the RNA flow from 1 up to 3 weeks back
    for i in range(7,22):
        Province_data["RNA_Flow_lag_{}".format(i)] = Province_data.RNA_Flow.shift(i)
    
    return Province_data
    

for province in data['Province'].unique():
    Province_data = data['Province']==province
    Province_data=data[Province_data]
    
    Province_data = add_lags(Province_data)
    
    data_lags=data_lags.append(Province_data)
    
# now clean up the data to perform the regressions
data_lags=data_lags.drop(columns=['RNA_Flow'])
data_lags=data_lags.drop(columns=['Total_reported'])
data_lags=data_lags.drop(columns=['Province'])

# sort data based on dates and remove rows with nan entries
data_lags=data_lags.sort_values(by=['Date'])
data_lags=data_lags.dropna()
data_lags=data_lags.reset_index(drop=True)

#now devide train data and test data
start_date=data_lags['Date'].iloc[0]
end_date=data_lags['Date'].iloc[len(data_lags['Date'])-1]
all_dates = pd.date_range(start=start_date,end=end_date) 
data_lags=data_lags.drop(['Date'],axis=1)

# Define a function for splitting the data into train and test data
def timeseries_train_test_split(X, y, test_size):
    """
        Perform train-test split with respect to time series structure
    """
    
    # get the index after which test set starts
    test_index = 181-21
    #times the number of provinces
    test_index = 12*test_index
    
    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]
    
    # set test index back
    test_index = test_index/12
    test_index = int(test_index)
    
    return test_index, X_train, X_test, y_train, y_test

y = data_lags.Hospital_admission
X = data_lags.drop(['Hospital_admission'], axis=1)


test_index, X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)


def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

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
    
    

# now first scale and perform the linear regressions
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

plotCoefficients(lr)


# for lasso and ridge make use of time series cross-validation
# for time-series cross-validation set 5 folds 
tscv = TimeSeriesSplit(n_splits=5)

ridge = RidgeCV(cv=tscv)
ridge.fit(X_train_scaled, y_train)
plotCoefficients(ridge)


lasso = LassoCV(cv=tscv)
lasso.fit(X_train_scaled, y_train)
plotCoefficients(lasso)


# Now make predictions for each seperate province based on these coefficients 

# first define the function for making plots
# Note that x_test must be the test data per province
def plotModelResults(model, X_train=X_train, X_test=X_test):
    """
        Plots modelled vs fact values, prediction intervals and anomalies
    
    """
    
    prediction = model.predict(X_test)
    
    plt.figure(figsize=(15, 7))
    plt.plot(prediction, "g", label="prediction", linewidth=2.0)
    plt.plot(y_test.values, label="actual", linewidth=2.0)
    
    
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
        

    
    error = mean_absolute_percentage_error(prediction, y_test)
    error = round(error, 2)
    if province == "NL":
        plt.title("Netherlands, Mean absolute percentage error {}%  Model:{}".format(error,model))
    else:
        plt.title("Province:{} Mean absolute percentage error {}%  Model:{}".format(province, error,model))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True);
    
# Now make predictions for each province in their test set
for province in data['Province'].unique():
    Province_data = data['Province']==province
    Province_data=data[Province_data]
    
    Province_data = add_lags(Province_data)
    Province_data.index = Province_data['Date']
    Province_data = Province_data.drop(columns=['Date'])
    Province_data =  Province_data.drop(columns=['Province'])
    Province_data = Province_data.drop(columns=['RNA_Flow'])
    Province_data = Province_data.drop(columns=['Total_reported'])
    
    # define the test data
    y = Province_data.dropna().Hospital_admission
    X = Province_data.dropna().drop(['Hospital_admission'], axis=1)
    
    X_test = X.iloc[test_index+7:]
    y_test = y.iloc[test_index+7:]
    X_test_scaled = scaler.transform(X_test)
    plotModelResults(lr, X_train=X_train_scaled, X_test=X_test_scaled)
    plotModelResults(ridge, X_train=X_train_scaled, X_test=X_test_scaled)
    plotModelResults(lasso, X_train=X_train_scaled, X_test=X_test_scaled)
    
#look how the model performs on national level
#first delog RNA_Flow
province = "NL"
data['RNA_Flow'] = np.exp(data['RNA_Flow'])

data=data.groupby('Date').agg({'RNA_Flow':'sum',
                               'Hospital_admission':'sum',
                               'Total_reported':'sum'})

# add dummies for days of the week
week_days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
for i in range(0,6):
    data[week_days[i]] = (data.index.weekday == i).astype(int)

# now take log of RNA flow again
data['RNA_Flow'] = np.log(data['RNA_Flow'])


data = add_lags(data)
data = data.drop(columns=['RNA_Flow'])
data = data.drop(columns=['Total_reported'])

# define the test data
y = data.dropna().Hospital_admission
X = data.dropna().drop(['Hospital_admission'], axis=1)

X_test = X.iloc[test_index+7:]
y_test = y.iloc[test_index+7:]
X_test_scaled = scaler.transform(X_test)
plotModelResults(lr, X_train=X_train_scaled, X_test=X_test_scaled)
plotModelResults(ridge, X_train=X_train_scaled, X_test=X_test_scaled)
plotModelResults(lasso, X_train=X_train_scaled, X_test=X_test_scaled)
    
coefficients_lasso = pd.Series(lasso.coef_,index=X_train.columns.values)
coefficients_lasso.to_csv('lasso_coef.csv',header=False)    
coefficients_ridge = pd.Series(ridge.coef_,index=X_train.columns.values)
coefficients_ridge.to_csv('ridge_coef.csv',header=False)
coefficients_lr = pd.Series(lr.coef_,index=X_train.columns.values)
coefficients_lr.to_csv('lr_coef.csv',header=False)




# check when training on the netherlands as a whole rather than per province
data=data.dropna()
y = data.Hospital_admission
X = data.drop(['Hospital_admission'], axis=1)

X_train = X.iloc[:test_index]
y_train = y.iloc[:test_index]
X_test = X.iloc[test_index+7:]
y_test = y.iloc[test_index+7:]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

plotCoefficients(lr)


# for lasso and ridge make use of time series cross-validation
# for time-series cross-validation set 5 folds 
tscv = TimeSeriesSplit(n_splits=5)

ridge = RidgeCV(cv=tscv)
ridge.fit(X_train_scaled, y_train)
plotCoefficients(ridge)


lasso = LassoCV(cv=tscv)
lasso.fit(X_train_scaled, y_train)
plotCoefficients(lasso)

plotModelResults(lr, X_train=X_train_scaled, X_test=X_test_scaled)
plotModelResults(ridge, X_train=X_train_scaled, X_test=X_test_scaled)
plotModelResults(lasso, X_train=X_train_scaled, X_test=X_test_scaled)


    
    

