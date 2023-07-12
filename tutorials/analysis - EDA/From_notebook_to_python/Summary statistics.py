#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Read-in-data" data-toc-modified-id="Read-in-data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Read-in data</a></span></li><li><span><a href="#Quick-checks" data-toc-modified-id="Quick-checks-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Quick checks</a></span></li><li><span><a href="#Summary-Statistics" data-toc-modified-id="Summary-Statistics-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Summary Statistics</a></span></li><li><span><a href="#Changes-Over-Time" data-toc-modified-id="Changes-Over-Time-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Changes Over Time</a></span></li><li><span><a href="#Resampling" data-toc-modified-id="Resampling-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Resampling</a></span><ul class="toc-item"><li><span><a href="#Avoiding-Foresight-Bias" data-toc-modified-id="Avoiding-Foresight-Bias-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>Avoiding Foresight Bias</a></span></li></ul></li><li><span><a href="#Rolling-Statistics" data-toc-modified-id="Rolling-Statistics-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Rolling Statistics</a></span></li><li><span><a href="#References" data-toc-modified-id="References-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>References</a></span></li><li><span><a href="#Requirements" data-toc-modified-id="Requirements-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Requirements</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Summary statistics
# 
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[1]:


import numpy as np
import pandas as pd
from pylab import mpl, plt
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")


# # Read-in data
# <hr style = "border:2px solid black" ></hr>

# - The file contains end-of-day (EOD) data for different financial instruments as retrieved from the Thomson Reuters Eikon Data API.

# In[2]:


filename = '../data/tr_eikon_eod_data.csv'  


# In[3]:


f = open(filename, 'r')  
f.readlines()[:5]  


# In[4]:


data = pd.read_csv(filename,  
                   index_col=0, 
                   parse_dates=True)  


# # Quick checks
# <hr style = "border:2px solid black" ></hr>

# In[5]:


data.info()  


# In[6]:


data.head()  


# In[7]:


data.tail()  


# In[8]:


data.plot(figsize=(10, 12), subplots=True)


# - The data used is from the Thomson Reuters (TR) Eikon Data API. 
# - In the TR world symbols for financial instruments are called Reuters Instrument Codes (RICs). 
# - The financial instruments that the single RICs represent are:

# In[9]:


instruments = ['Apple Stock', 'Microsoft Stock',
               'Intel Stock', 'Amazon Stock', 'Goldman Sachs Stock',
               'SPDR S&P 500 ETF Trust', 'S&P 500 Index',
               'VIX Volatility Index', 'EUR/USD Exchange Rate',
               'Gold Price', 'VanEck Vectors Gold Miners ETF',
               'SPDR Gold Trust']


# In[10]:


for ric, name in zip(data.columns, instruments):
    print('{:8s} | {}'.format(ric, name))


# # Summary Statistics
# <hr style = "border:2px solid black" ></hr>

# In[11]:


data.info()  


# In[12]:


data.describe().round(2)  


# In[13]:


data.mean()  


# In[14]:


data.aggregate([min,  
                np.mean,  
                np.std,  
                np.median,  
                max]  
).round(2)


# # Changes Over Time
# <hr style = "border:2px solid black" ></hr>

# - Statistical analysis methods are often based on changes over time and not the absolute values themselves. 
# - There are multiple options to calculate the changes in a time series over time including:
#     - Absolute differences
#     - Percentage changes
#     - Logarithmic (log) returns

# In[15]:


data.diff().head()  


# In[16]:


data.diff().mean()  


# - From a statistics point of view, absolute changes are not optimal because they are dependent on the scale of the time series data itself. 
# - Therefore, percentage changes are usually preferred.

# In[17]:


data.pct_change().round(3).head()  


# In[18]:


data.pct_change().mean().plot(kind='bar', figsize=(10, 6));  
# plt.savefig('../../images/ch08/fts_02.png');


# - As an alternative to percentage returns, log returns can be used. In some scenarios, they are easier to handle and therefore often preferred in a financial context.
# - One of the advantages is additivity over time, which does not hold true for simple percentage changes/ returns.
# - The figure below shows the cumulative log returns for the single financial time series. This type of plot leads to some form of normalization.

# In[19]:


rets = np.log(data / data.shift(1))  


# In[20]:


rets.head().round(3)  


# In[21]:


rets.cumsum().apply(np.exp).plot(figsize=(10, 6));  


# # Resampling
# <hr style = "border:2px solid black" ></hr>

# - Resampling is an important operation on financial time series data. 
# - Usually this takes the form of downsampling, meaning that, for example, a tick data series is resampled to one-minute intervals or a time series with daily observations is resampled to one with weekly or monthly observations

# In[22]:


data.resample('1w', label='right').last().head()  


# In[23]:


data.resample('1m', label='right').last().head()  


# In[24]:


rets.cumsum().apply(np.exp). resample('1m', label='right').last(
                          ).plot(figsize=(10, 6));  


# ## Avoiding Foresight Bias

# <div class="alert alert-info">
# <font color=black>
# 
# - When resampling, pandas takes by default in many cases the left label (or index value) of the interval.
# - To be financially consistent, make sure to use the right label (index value) and in general the last available data point in the interval. 
# - Otherwise, a foresight bias might sneak into the financial analysis.
# - Foresight bias — or, in its strongest form, **perfect foresight**— means that at some point in the financial analysis, data is used that only becomes available at a later point. The result might be “too good” results, for example, when backtesting a trading strategy.
#     
# </font>
# </div>

# # Rolling Statistics
# <hr style = "border:2px solid black" ></hr>

# In[25]:


sym = 'AAPL.O'


# In[26]:


data = pd.DataFrame(data[sym]).dropna()


# In[27]:


data.tail()


# In[28]:


window = 20  


# In[29]:


data['min'] = data[sym].rolling(window=window).min()  


# In[30]:


data['mean'] = data[sym].rolling(window=window).mean()  


# In[31]:


data['std'] = data[sym].rolling(window=window).std()  


# In[32]:


data['median'] = data[sym].rolling(window=window).median()  


# In[33]:


data['max'] = data[sym].rolling(window=window).max()  


# In[34]:


data['ewma'] = data[sym].ewm(halflife=0.5, min_periods=window).mean()  


# In[35]:


data.dropna().head()


# In[36]:


ax = data[['min', 'mean', 'max']].iloc[-200:].plot(
    figsize=(10, 6), style=['g--', 'r--', 'g--'], lw=1)  
data[sym].iloc[-200:].plot(ax=ax, lw=2.0);  


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://github.com/yhilpisch/py4fi2nd/blob/master/code/ch08/08_financial_time_series.ipynb
# - Hilpisch, Yves. Python for finance: mastering data-driven finance. O'Reilly Media, 2018.
# - [Data](https://github.com/yhilpisch/py4fi2nd/tree/master/source)
# 
# </font>
# </div>

# # Requirements
# <hr style = "border:2px solid black" ></hr>

# In[37]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -iv')

