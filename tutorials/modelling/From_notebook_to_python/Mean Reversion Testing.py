#!/usr/bin/env python
# coding: utf-8

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
#
# **What?** Mean reversion testing
#
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[19]:


import pandas as pd
import statsmodels.tsa.stattools as ts
import pandas_datareader as pdr
import datetime as dt
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn


# # Random-walk
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# - The basic idea when trying to ascertain if a time series is mean-reverting is to use a statistical test to see if it differs from the behaviour of a random walk.
# - A **random walk** is a time series where the next directional movement is completely independent of any past movements - in essence the time series has no "memory" of where it has been.
# - A **mean-reverting** time series, however, is different. The change in the value of the time series in the next time period is proportional to the current value. Specifically, it is proportional to the difference between the mean historical price and the current price.
#
# </font>
# </div>

# # Ornstein-Uhlenbeck series
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# - Mathematically, such a (continuous) time series is referred to as an Ornstein-Uhlenbeck process. If we can show, statistically, that a price series behaves like an **Ornstein-Uhlenbeck series** then we can begin the process of forming a trading strategy around it.
# - Thus the goal of this chapter is to outline the statistical tests necessary to identify mean reversion and then use Python libraries (in particular statsmodels) in order to implement these tests. In particular, we will study the concept of stationarity and how to test for it.
#
# </font>
# </div>

#  ![image.png](attachment:image.png)

# # Stationary timeseries
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# - A time series (or stochastic process) is defined to be strongly stationary if its joint probability distribution is invariant under translations in time or space. In particular, and of key importance for traders, the mean and variance of the process do not change over time or space and they each do not follow a trend.
#
# - A critical feature of stationary price series is that the prices within the series diffuse from their initial value at a rate slower than that of a GBM.
#
# - By measuring the rate of this diffusive behaviour we can identify the nature of the time series and thus detect whether it is mean-reverting.
#
# - The **Hurst Exponent** helps us characterise the stationarity of a time series.
#
# </font>
# </div>

# # Get the data
# <hr style = "border:2px solid black" ></hr>

# In[9]:


# year-month-day format
start = dt.datetime(2004, 9, 1)
end = dt.datetime(2020, 8, 31)

goog_df = pdr.get_data_yahoo("GOOG", start, end)


# In[10]:


goog_df


# # Test for mean reversion
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# - There are two alternative methods for detecting mean-reversion:
#     - via the concept of stationarity
#     - via ADF test
#
# </font>
# </div>

# ## Augmented Dickey-Fuller (ADF) Test

# <div class="alert alert-info">
# <font color=black>
#
# - The ADF test makes use of the fact that if a price series possesses mean reversion, then the next price level will be proportional to the current price level.
# - Mathematically, the ADF is based on the idea of testing for the presence of a unit root in an autoregressive time series sample.
#
#
# </font>
# </div>

# In[17]:


adf = ts.adfuller(goog_df["Adj Close"], 1)
print(adf)


# <div class="alert alert-info">
# <font color=black>
#
# - Mean hypothesis: the series behavies like a random walk.
# - Since the calculated value of the test statistic is larger than any of the critical values at the 1, 5 or 10 percent levels, we cannot reject the null hypothesis and thus we are unlikely to have found a mean reverting time series.
# - This is in line with our tuition as most equities behave akin to Geometric Brownian Motion (GBM), i.e. a random walk.
#
# </font>
# </div>

# ## Stationary - Hurst exponent

# <div class="alert alert-info">
# <font color=black>
#
# - The goal of the Hurst Exponent is to provide us with a scalar value that will help us to identify (within the limits of statistical estimation) whether a series is mean reverting, random walking or trending:
#
#     - H < 0.5 - The time series is mean reverting
#     - H = 0.5 - The time series is a Geometric Brownian Motion
#     - H > 0.5 - The time series is trending
#
# </font>
# </div>

# In[ ]:


def hurst(ts):
    """
    Returns the Hurst Exponent of the time series vector ts

    Parameters
    ----------
    ts : `numpy.array`
        Time series upon which the Hurst Exponent will be calculated

    Returns
    -------
    'float'
        The Hurst Exponent from the poly fit output
    """
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0] * 2.0


# In[20]:


# Create a synthetic Gometric Brownian Motion, Mean-Reverting and Trending Series
gbm = log(cumsum(randn(100000)) + 1000)
mr = log(randn(100000) + 1000)
tr = log(cumsum(randn(100000) + 1) + 1000)


# In[21]:


# Output the Hurst Exponent for each of the above series
# and the price of Google (the Adjusted Close price) for
# the ADF test given above in the article
print("Hurst(GBM):   %s" % hurst(gbm))
print("Hurst(MR):    %s" % hurst(mr))
print("Hurst(TR):    %s" % hurst(tr))


# In[22]:


# Assuming you have run the above code to obtain 'goog'!
print("Hurst(GOOG):  %s" % hurst(goog_df["Adj Close"].values))


# <div class="alert alert-info">
# <font color=black>
#
# - `GOOG` is close to 0.5 indicating that it is similar to a GBM.
#
# </font>
# </div>

# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
#
# - https://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing/
# - https://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing-Part-II/
#
# </font>
# </div>

# In[ ]:
