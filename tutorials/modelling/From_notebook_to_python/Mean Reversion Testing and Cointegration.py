#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Random-walk" data-toc-modified-id="Random-walk-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Random-walk</a></span></li><li><span><a href="#Ornstein-Uhlenbeck-series" data-toc-modified-id="Ornstein-Uhlenbeck-series-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Ornstein-Uhlenbeck series</a></span></li><li><span><a href="#Stationary-timeseries" data-toc-modified-id="Stationary-timeseries-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Stationary timeseries</a></span></li><li><span><a href="#Cointegration" data-toc-modified-id="Cointegration-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Cointegration</a></span></li><li><span><a href="#Get-the-data" data-toc-modified-id="Get-the-data-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Get the data</a></span></li><li><span><a href="#EDA" data-toc-modified-id="EDA-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>EDA</a></span></li><li><span><a href="#Create-residual" data-toc-modified-id="Create-residual-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Create residual</a></span></li><li><span><a href="#Cointegrated-Augmented-Dickey-Fuller-(CADF)" data-toc-modified-id="Cointegrated-Augmented-Dickey-Fuller-(CADF)-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Cointegrated Augmented Dickey-Fuller (CADF)</a></span></li><li><span><a href="#References" data-toc-modified-id="References-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Mean reversion testing and cointegration
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

rcParams['figure.figsize'] = 17, 8
rcParams['font.size'] = 20


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

# # Get the data
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#  
# - Let's consider two energy sector equities Exxon Mobil Corp given by the ticker XOM and United States Oil Fund given by the ticker USO.
#     
# - Both are exposed to similar market conditions and thus will likely have a stationary pairs relationship. 
# 
# </font>
# </div>

# In[2]:


# year-month-day format
start = dt.datetime(2019, 1, 1) 
end = dt.datetime(2020, 1, 1)

xom = pdr.get_data_yahoo("XOM", start, end) 
uso = pdr.get_data_yahoo("USO", start, end)


# In[3]:


xom


# In[4]:


uso


# In[5]:


df = pd.DataFrame(index=xom.index)
df["XOM"] = xom["Adj Close"]
df["USO"] = uso["Adj Close"]


# In[6]:


df


# # EDA
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#  
# - By creating a scatter plot of their prices, we'll see the relationship is broadly linear.
# 
# </font>
# </div>

# In[7]:


fig = df.plot(title="USO and XOM Daily Prices")
fig.set_ylabel("Price($)")
plt.show()


# In[8]:


df.plot.scatter(x=0, y=1, title="USO and XOM Price Scatterplot")
plt.show()


# # Create residual
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#  
# - The pairs trade essentially works by using a linear model for a relationship between the two stock prices.
# - While plotting the residual is does not seem particularly stationary.
#     
# </font>
# </div>

# In[9]:



# Create OLS model
Y = df['USO']
x = df['XOM']
x = sm.add_constant(x)
model = sm.OLS(Y, x)
res = model.fit()

# Beta hedge ratio (coefficent from OLS)
beta_hr = res.params[1]
print(f'Beta Hedge Ratio: {beta_hr}')

# Residuals
df["Residuals"] = res.resid


# In[10]:


df


# In[11]:


df.index


# In[12]:


df.plot(y="Residuals")
fig.set_ylabel("Price($)")
plt.show()


# # Cointegrated Augmented Dickey-Fuller (CADF)
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - The Cointegrated Augmented Dickey-Fuller (CADF) determines the optimal hedge ratio by performing a linear regression against the two time series and then tests for stationarity under the linear combination.
# - It can be seen that the calculated test statistic of -2.891 is more negative than the 5% critical value of -2.873, which means that we can reject the null hypothesis that there isn't a cointegrating relationship at the 5% level. 
# - Hence we can conclude, with a reasonable degree of certainty, that USO and XOM possess a cointegrating relationship, at least for the time period sample considered.
# 
# </font>
# </div>

# In[13]:


ts.adfuller(df["Residuals"])


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




