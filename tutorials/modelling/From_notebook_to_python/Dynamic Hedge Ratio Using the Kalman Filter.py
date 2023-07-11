#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Cointegration" data-toc-modified-id="Cointegration-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Cointegration</a></span></li><li><span><a href="#Dynamic-Hedge-Ratio" data-toc-modified-id="Dynamic-Hedge-Ratio-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Dynamic Hedge Ratio</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Get-the-data" data-toc-modified-id="Get-the-data-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Get the data</a></span></li><li><span><a href="#EDA" data-toc-modified-id="EDA-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>EDA</a></span></li><li><span><a href="#Kalman-filter" data-toc-modified-id="Kalman-filter-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Kalman filter</a></span></li><li><span><a href="#Post-processing" data-toc-modified-id="Post-processing-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Post-processing</a></span></li><li><span><a href="#Conclusions" data-toc-modified-id="Conclusions-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Conclusions</a></span></li><li><span><a href="#References" data-toc-modified-id="References-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
#
# **What?** Dynamic Hedge Ratio Using the Kalman Filter
#
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[1]:


import pandas as pd
import statsmodels.tsa.stattools as ts
import pandas_datareader as pdr
import datetime as dt
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from matplotlib import rcParams

rcParams["figure.figsize"] = 17, 8
rcParams["font.size"] = 20


# # Cointegration
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# - It is difficult to find mean-reverting asset as equities broadly behave like GBMs and hence render the mean-reverting trade strategies relatively useless.
# - However, we can still create a portfolio of price series that is stationary so we can apply a mean-reverting trading strategies.
# - The **pairs trade** does exactly this. Two companies in the same sector are likely to be exposed to similar market factors, which affect their businesses. Occasionally their relative stock prices will diverge due to certain events, but will revert to the long-running mean.
#
# </font>
# </div>

# # Dynamic Hedge Ratio
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# - A common quant trading technique involves taking two assets that form a cointegrating relationship and utilising a mean-reverting approach to construct a trading strategy. This can be carried out by performing a linear regression between the two assets.
#
# - Any parameters introduced via this structural relationship such as the hedging ratio between the two assets are likely to be time-varying, thus we need to find a way for adjusting the hedging ratio over time.
#
# - One approach to this problem is to utilise a rolling linear regression with a lookback window for which the lookback window length must be found often via cross-validation.
#
# - A more sophisticated approach is to utilise a state space model that treats the "true" hedge ratio as an unobserved hidden variable and attempts to estimate it with "noisy" observations. In our case this means the pricing data of each asset. The **Kalman filter** performs exactly this task.
#
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from pykalman import KalmanFilter


# # Get the data
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# - We are going to consider two fixed income ETFs, namely the iShares 20+ Year Treasury Bond ETF (TLT) and the iShares 3-7 Year Treasury Bond ETF (IEI).
# - Both ETFs track the performance of varying duration US Treasury bonds and as such are both exposed to similar market factors.
#
# </font>
# </div>

# In[3]:


# Choose the ETF symbols to work with along with
# start and end dates for the price histories
etfs = ["TLT", "IEI"]
start_date = "2010-8-01"
end_date = "2016-08-01"


# In[4]:


# Obtain the adjusted closing prices from Yahoo finance
etf_df1 = pdr.get_data_yahoo(etfs[0], start_date, end_date)
etf_df2 = pdr.get_data_yahoo(etfs[1], start_date, end_date)


# In[5]:


prices = pd.DataFrame(index=etf_df1.index)
prices[etfs[0]] = etf_df1["Adj Close"]
prices[etfs[1]] = etf_df2["Adj Close"]


# In[9]:


prices


# # EDA
# <hr style = "border:2px solid black" ></hr>

# In[6]:


"""
Create a scatterplot of the two ETF prices, which is
coloured by the date of the price to indicate the
changing relationship between the sets of prices
"""

# Create a yellow-to-red colourmap where yellow indicates
# early dates and red indicates later dates
plen = len(prices)
colour_map = plt.cm.get_cmap("YlOrRd")
colours = np.linspace(0.1, 1, plen)
# Create the scatterplot object
scatterplot = plt.scatter(
    prices[etfs[0]],
    prices[etfs[1]],
    s=30,
    c=colours,
    cmap=colour_map,
    edgecolor="k",
    alpha=0.8,
)
# Add a colour bar for the date colouring and set the
# corresponding axis tick labels to equal string-formatted dates
colourbar = plt.colorbar(scatterplot)
colourbar.ax.set_yticklabels([str(p.date()) for p in prices[:: plen // 9].index])
plt.xlabel(prices.columns[0])
plt.ylabel(prices.columns[1])
plt.show()


# # Kalman filter
# <hr style = "border:2px solid black" ></hr>

# In[7]:


"""
Utilise the Kalman Filter from the PyKalman package
to calculate the slope and intercept of the regressed
ETF prices.
"""

delta = 1e-5
trans_cov = delta / (1 - delta) * np.eye(2)
obs_mat = np.vstack([prices[etfs[0]], np.ones(prices[etfs[0]].shape)]).T[:, np.newaxis]
kf = KalmanFilter(
    n_dim_obs=1,
    n_dim_state=2,
    initial_state_mean=np.zeros(2),
    initial_state_covariance=np.ones((2, 2)),
    transition_matrices=np.eye(2),
    observation_matrices=obs_mat,
    observation_covariance=1.0,
    transition_covariance=trans_cov,
)
state_means, state_covs = kf.filter(prices[etfs[1]].values)


# # Post-processing
# <hr style = "border:2px solid black" ></hr>

# In[8]:


# Plot the slope and intercept changes from the Kalman Filter calculated values.

pd.DataFrame(
    dict(slope=state_means[:, 0], intercept=state_means[:, 1]), index=prices.index
).plot(subplots=True)
plt.show()


# # Conclusions
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-danger">
# <font color=black>
#
# - We have mentioned how using a rolling mean would have required to perform a cross-validated procedure to find the window.
# - **Be aware that**, if this was to be put into production as a live trading strategy it would be necessary to
# optimise the delta parameter across baskets of pairs of ETFs utilising cross-validation.
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
# - https://www.quantstart.com/advanced-algorithmic-trading-ebook/
#
# </font>
# </div>

# In[ ]:
