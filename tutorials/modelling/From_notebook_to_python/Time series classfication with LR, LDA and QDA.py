#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Project-goal" data-toc-modified-id="Project-goal-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Project goal</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Time Series classficatio with LR, LDA and QDA
# 
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[29]:


import datetime
import numpy as np
import pandas as pd
import sklearn

#from pandas.io.data import DataReader
#from pandas_datareader import data as DataReader
import pandas_datareader.data as web
from sklearn.linear_model import LogisticRegression
#from sklearn.lda import LDA
#from sklearn.qda import QDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA


# In[30]:


# Getting rid of the warning messages
import warnings
warnings.filterwarnings("ignore")


# # Project goal
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - Build a classifiers to predict the direction of the closing price at day based solely on price information known at day . An upward directional move means that the closing price at is higher than the price at opening, while a downward move implies a closing price at lower than at opening.
# 
# - If we can determine the direction of movement in a manner that significantly exceeds a 50% hit rate, with low error and a good statistical significance, then we are on the road to forming a basic systematic trading strategy based on our forecasts. 
# 
# - At this stage we're not concerned with the most up to date machine learning classification algorithms.
# 
# </font>
# </div>

# In[31]:


def create_lagged_series(symbol, start_date, end_date, lags=5):
    """This creates a pandas DataFrame that stores the percentage returns of the 
    adjusted closing value of a stock obtained from Yahoo Finance, along with 
    a number of lagged returns from the prior trading days (lags defaults to 5 days).
    Trading volume, as well as the Direction from the previous day, are also included."""

    # Obtain stock information from Yahoo Finance
    ts = web.DataReader(symbol, "yahoo", start_date -
                    datetime.timedelta(days=365), end_date)

    # Create the new lagged DataFrame
    tslag = pd.DataFrame(index=ts.index)
    tslag["Today"] = ts["Adj Close"]
    tslag["Volume"] = ts["Volume"]

    # Create the shifted lag series of prior trading period close values
    for i in range(0, lags):
        tslag["Lag%s" % str(i+1)] = ts["Adj Close"].shift(i+1)

    # Create the returns DataFrame
    tsret = pd.DataFrame(index=tslag.index)
    tsret["Volume"] = tslag["Volume"]
    tsret["Today"] = tslag["Today"].pct_change()*100.0

    # If any of the values of percentage returns equal zero, set them to
    # a small number (stops issues with QDA model in scikit-learn)
    for i, x in enumerate(tsret["Today"]):
        if (abs(x) < 0.0001):
            tsret["Today"][i] = 0.0001

    # Create the lagged percentage returns columns
    for i in range(0, lags):
        tsret["Lag%s" % str(i+1)] = tslag["Lag%s" %
                                          str(i+1)].pct_change()*100.0

    # Create the "Direction" column (+1 or -1) indicating an up/down day
    tsret["Direction"] = np.sign(tsret["Today"])
    tsret = tsret[tsret.index >= start_date]

    return tsret


# In[32]:


def fit_model(name, model, X_train, y_train, X_test, pred):
    """Fits a classification model (for our purposes this is LR, LDA and QDA)
    using the training data, then makes a prediction and subsequent "hit rate"
    for the test data."""

    # Fit and predict the model on the training, and then test, data
    model.fit(X_train, y_train)
    pred[name] = model.predict(X_test)

    # Create a series with 1 being correct direction, 0 being wrong
    # and then calculate the hit rate based on the actual direction
    pred["%s_Correct" % name] = (1.0+pred[name]*pred["Actual"])/2.0
    hit_rate = np.mean(pred["%s_Correct" % name])
    print("%s: %.3f" % (name, hit_rate))


# In[33]:


# Create a lagged series of the S&P500 US stock market index
snpret = create_lagged_series("^GSPC", datetime.datetime(2001,1,10), datetime.datetime(2005,12,31), lags=5)

# Use the prior two days of returns as predictor values, with direction as the response
X = snpret[["Lag1","Lag2"]]
y = snpret["Direction"]

# The test data is split into two parts: Before and after 1st Jan 2005.
start_test = datetime.datetime(2005,1,1)

# Create training and test sets
X_train = X[X.index < start_test]
X_test = X[X.index >= start_test]
y_train = y[y.index < start_test]
y_test = y[y.index >= start_test]

# Create prediction DataFrame
pred = pd.DataFrame(index=y_test.index)
pred["Actual"] = y_test

# Create and fit the three models    
print("Hit Rates:")
models = [("LR", LogisticRegression()), ("LDA", LDA()), ("QDA", QDA())]
for m in models:
    fit_model(m[0], m[1], X_train, y_train, X_test, pred)


# - https://www.quantstart.com/articles/Forecasting-Financial-Time-Series-Part-1/

# In[ ]:




