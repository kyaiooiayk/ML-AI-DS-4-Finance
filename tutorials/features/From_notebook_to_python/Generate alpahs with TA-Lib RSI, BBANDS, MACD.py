#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Load-data" data-toc-modified-id="Load-data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Load data</a></span></li><li><span><a href="#Features-engineering---indicators" data-toc-modified-id="Features-engineering---indicators-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Features engineering - indicators</a></span></li><li><span><a href="#References" data-toc-modified-id="References-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - **What?** SVM buy or sell strategy
#     
# </font>
# </div>

# # Imports
# <hr style="border:2px solid black"> </hr>

# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#!pip install TA-lib
#!conda install -c conda-forge ta-lib
import talib as ta


# In[3]:


from talib import RSI, BBANDS, MACD


# # Load data
# <hr style="border:2px solid black"> </hr>

# In[22]:


data = pd.read_csv('../datasource/random_stock_data.csv')
data.head(30)


# # Features engineering - indicators
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-info">
# <font color=black>
#     
# - **Relative Strength Index (RSI)** The RSI provides technical traders with signals about bullish and bearish price momentum, and it is often plotted beneath the graph of an asset’s price.
# - **Bollinger Bands** are envelopes plotted at a standard deviation level above and below a simple moving average of the price. Because the distance of the bands is based on standard deviation, they adjust to volatility swings in the underlying price. 
# 
# </font>
# </div>

# In[24]:


# Compute Bollinger Bands
up, mid, low = BBANDS(data.Close, timeperiod=21,
                      nbdevup=2, nbdevdn=2, matype=0)


# In[25]:


# Compute Relative Strength Index
rsi = RSI(data.Close, timeperiod=14)


# <div class="alert alert-info">
# <font color=black>
#     
# - The MACD computes the difference between two Exponential Moving Averages (EMA), one longer- and one shorter-term.
# - The ta-lib MACD Indicator implementation has four inputs:
#     - the close price
#     - fastperiod: the short-term EMA period
#     - slowperiod: the long-term EMA period
#     - signalperiod: the period for the EMA of the MACD itself
# - It has three outputs:
#     - macd is the difference between the fast EMA and slow EMA.
#     - macdsignal is the EMA of the MACD value with period signalperiod
#     - macdhist computes the difference between macd and macdsignal
# 
# </font>
# </div>
# 

# In[26]:


macd, macdsignal, macdhist = MACD(
    data.Close, fastperiod=12, slowperiod=26, signalperiod=9)


# In[30]:


macd_data = pd.DataFrame({'AAPL': data.Close, 'MACD': macd,
                         'MACD Signal': macdsignal, 'MACD History': macdhist})

fig, axes = plt.subplots(nrows=2, figsize=(15, 8))
macd_data.AAPL.plot(ax=axes[0])
macd_data.drop('AAPL', axis=1).plot(ax=axes[1])
fig.tight_layout()
sns.despine()


# In[31]:


data = pd.DataFrame({'AAPL': data.Close, 'BB Up': up,
                    'BB Mid': mid, 'BB down': low, 'RSI': rsi, 'MACD': macd})


# In[32]:


fig, axes= plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
data.drop(['RSI', 'MACD'], axis=1).plot(ax=axes[0], lw=1, title='Bollinger Bands')
data['RSI'].plot(ax=axes[1], lw=1, title='Relative Strength Index')
axes[1].axhline(70, lw=1, ls='--', c='k')
axes[1].axhline(30, lw=1, ls='--', c='k')
data.MACD.plot(ax=axes[2], lw=1, title='Moving Average Convergence/Divergence', rot=0)
axes[2].set_xlabel('')
fig.tight_layout()
sns.despine();


# # References
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - [Dataset](https://github.com/Datatouille/findalpha/tree/master)
# - [Tutorial](https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/04_alpha_factor_research/02_how_to_use_talib.ipynb)
# - [https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/bollinger-bands](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/bollinger-bands)
#     
# </font>
# </div>

# In[ ]:




