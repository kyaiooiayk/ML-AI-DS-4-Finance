#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Import-dataset" data-toc-modified-id="Import-dataset-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Import dataset</a></span></li><li><span><a href="#Kalman-Filter" data-toc-modified-id="Kalman-Filter-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Kalman Filter</a></span></li><li><span><a href="#Random-walk" data-toc-modified-id="Random-walk-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Random walk</a></span></li><li><span><a href="#Connection-btw-Random-walk-and-Karman-filter" data-toc-modified-id="Connection-btw-Random-walk-and-Karman-filter-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Connection btw Random walk and Karman filter</a></span><ul class="toc-item"><li><span><a href="#Chose-filter-parameters" data-toc-modified-id="Chose-filter-parameters-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Chose filter parameters</a></span></li><li><span><a href="#Applying-the-filter-to-stock" data-toc-modified-id="Applying-the-filter-to-stock-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Applying the filter to stock</a></span></li></ul></li><li><span><a href="#References" data-toc-modified-id="References-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Karman filter
# 
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[6]:


import pandas_datareader as pdr
import datetime as dt
from datetime import datetime
import itertools

import pandas as pd
import pandas_datareader.data as web
from pykalman import KalmanFilter
import pywt

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
idx = pd.IndexSlice


# # Import dataset
# <hr style = "border:2px solid black" ></hr>

# In[95]:


# NLFLX is the netflix stock price
start = dt.datetime(2020, 1, 1)
data = pdr.get_data_yahoo("NFLX", start)


# In[96]:


data.head()


# In[97]:


df = data["Adj Close"]
df


# In[99]:


fig, ax = plt.subplots()
df.plot(ax=ax)
ax.set_ylabel("ATR")
data.plot
ax.legend()


# # Kalman Filter 
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - The Kalman filter is a dynamic linear model of sequential data like a time series that adapts to new information as it arrives. Rather than using a fixed-size window like a moving average or a given set of weights like an exponential moving average, it incorporates new data into its estimates of the current value of the time series based on a probabilistic model.
# 
# - **Why Use the Word “Filter”?** A noisy time series is like a time series with many rough edges. The Kalman Filter estimates the underlying states. It “filters” out the rough edges to reveal relatively smooth patterns.
# 
# - Notice that **smoothing** is different from the smoothing of moving average methods. The moving average methods take the past points, even errors, with differential weights to get a smoothed line. In contrast, the Kalman Filter recognizes some data points as noise. 
# 
# - A Kalman filter is called an optimal estimator. Optimal in what sense? The Kalman filter minimizes the mean square error of the estimated parameters. So it is the best-unbiased estimator.
# 
# - It is recursive so that `Xt+1` can be calculated only with `Xt`. and does not require the presence of all past data points `X0, X1, …, Xt`. This is an important merit for real-time processing.
#     
# - The error terms in Equations (1) and (2) are both Gaussian distributions, so the error term in the predicted values also follows the Gaussian distribution.
#     
# - There is no need to provide labeled target data to “train” a model.
#     
# </font>
# </div>

# # Random walk
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - Stock price movement is widely modeled as a **random walk**. It means at each point in time the series merely takes a random step away from its last position, with steps that the mean value is zero. 
# 
# - This is described in Equation (1) describes: `Xt = At * Xt-1`, where At is the transition matrix and At = 1. This is also called random-walk-without-drift. 
# 
# - If the mean step size is a nonzero value α, it is called random-walk-with-drift. It becomes `Xt = At * Xt-1 + α`. 
#         
# </font>
# </div>

# # Connection btw Random walk and Karman filter
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - It is straightforward to implement the Kalman Filter. Equation (1) only has one term Xt-1. So the transition matrix At only has one value [1.0]. It is 1.0 because of the random walk assumption. We also allow it to have small turbulence of 0.01, meaning the At can sometimes go above 1.0 and sometimes go below 1.0. It is denoted by transition_covariance=0.01. The observation matrix Ct also is 1.0.
# 
# - The initial value `X0` can be any value as you'll see how it quickly converges to the true value. The error terms qt and rt are Gaussian-distributed with a mean of 0 and variance of 1.0. So initial_state_covariance=1 and observation_covariance=1.
# 
# </font>
# </div>

# ## Chose filter parameters
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - `transition_matrices = [1]` = The value for At. It is a random walk so is set to 1.0
# - `observation_matrices = [1]` =  The value for Ht.
# - `initial_state_mean = 0` = Any initial value. It will converge to the true state value.
# - `initial_state_covariance = 1` = Sigma value for the Qt in Equation (1) the Gaussian distribution
# - `observation_covariance=1` = Sigma value for the Rt in Equation (2) the Gaussian distribution
# - `transition_covariance=.01` = A small turbulence in the random walk parameter 1.0
# 
# </font>
# </div>

# In[100]:


kf = KalmanFilter(transition_matrices = [1],
                  observation_matrices = [1],
                  initial_state_mean = 0,
                  initial_state_covariance = 1,
                  observation_covariance=1,
                  transition_covariance=.01)


# ## Applying the filter to stock
# <hr style = "border:2px solid black" ></hr>

# In[101]:


# Estimate the hidden state
state_means, _ = kf.filter(df)


# In[102]:


Add it to the dataset
data['KF'] = np.array(state_means)


# In[103]:


data


# In[106]:


fig, ax = plt.subplots()
df.plot(ax=ax)
data["KF"].plot(ax=ax)
data.plot
ax.legend()


# <div class="alert alert-info">
# <font color=black>
# 
# - **The first few estimates are far off from the target “Adj Close”** However, it converges to the target “Adj Close” after the first few observations, as shown in the plot below.
#     
# - **We do not set up a training dataset to train the model** The Kalman Filter does not work that way. The purpose of training a model is to get the parameters At. The Kalman Filter gets a parameter value for each new time step t.
# 
# </font>
# </div>

# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://medium.com/dataman-in-ai/kalman-filter-explained-4d65b47916bf
# - https://www.quantstart.com/articles/State-Space-Models-and-the-Kalman-Filter/
#     
# </font>
# </div>

# In[ ]:




