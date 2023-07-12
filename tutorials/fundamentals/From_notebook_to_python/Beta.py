#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Read-in-data" data-toc-modified-id="Read-in-data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Read-in data</a></span></li><li><span><a href="#Quick-checks" data-toc-modified-id="Quick-checks-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Quick checks</a></span></li><li><span><a href="#Summary-Statistics" data-toc-modified-id="Summary-Statistics-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Summary Statistics</a></span></li><li><span><a href="#Clean-data" data-toc-modified-id="Clean-data-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Clean data</a></span></li><li><span><a href="#Calculating-Beta" data-toc-modified-id="Calculating-Beta-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Calculating Beta</a></span><ul class="toc-item"><li><span><a href="#via-OLS" data-toc-modified-id="via-OLS-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>via OLS</a></span></li><li><span><a href="#via-variance" data-toc-modified-id="via-variance-7.2"><span class="toc-item-num">7.2&nbsp;&nbsp;</span>via variance</a></span></li></ul></li><li><span><a href="#References" data-toc-modified-id="References-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>References</a></span></li><li><span><a href="#Requirements" data-toc-modified-id="Requirements-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Requirements</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Beta
# 
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[32]:


import numpy as np
import pandas as pd
from pylab import mpl, plt
import statsmodels.api as sm


# In[ ]:


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


# In[7]:


data.head()  


# In[9]:


data.tail()  


# In[10]:


data.plot(figsize=(10, 12), subplots=True)


# - The data used is from the Thomson Reuters (TR) Eikon Data API. 
# - In the TR world symbols for financial instruments are called Reuters Instrument Codes (RICs). 
# - The financial instruments that the single RICs represent are:

# In[11]:


instruments = ['Apple Stock', 'Microsoft Stock',
               'Intel Stock', 'Amazon Stock', 'Goldman Sachs Stock',
               'SPDR S&P 500 ETF Trust', 'S&P 500 Index',
               'VIX Volatility Index', 'EUR/USD Exchange Rate',
               'Gold Price', 'VanEck Vectors Gold Miners ETF',
               'SPDR Gold Trust']


# In[12]:


for ric, name in zip(data.columns, instruments):
    print('{:8s} | {}'.format(ric, name))


# # Summary Statistics
# <hr style = "border:2px solid black" ></hr>

# In[13]:


data.info()  


# In[14]:


data.describe().round(2)  


# In[18]:


data.mean()  


# In[19]:


data.aggregate([min,  
                np.mean,  
                np.std,  
                np.median,  
                max]  
).round(2)


# # Clean data
# <hr style = "border:2px solid black" ></hr>

# In[22]:


data = data.dropna()
data


# In[26]:


data[['AMZN.O', 'SPY']].plot(
    figsize=(10, 6), style=['g--', 'r--', 'g--'], lw=1);


# # Calculating Beta
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - A stock that swings more than the market over time has a beta greater than 1.0 = High Beta
# - A stock beta is less than 1.0 = Low Beta
# - High-Beta tend to be Riskier but provide the potential for higher returns; Low-Beta stocks pose less risk but typically yield lower returns.
# - A stock beta 1.0 = Mr. Market
# - For beta to be meaningful, the stock should be related to the benchmark that is used in the calculation.
# - The S&P 500 has a beta of 1.0.
# Stocks with betas above 1 will tend to move with more momentum than the S&P 500; stocks with betas less than 1 with less momentum.
# 
# </font>
# </div>

# ## via OLS

# In[35]:


# Create a regression model
reg = sm.OLS(data['AMZN.O'], data['SPY'])


# In[36]:


# Fit the model
results = reg.fit()


# In[38]:


results.params


# In[39]:


# Print Beta of Amazon
print(f"Beta for Amazon {results.params[0]}")


# ## via variance

# In[42]:


# Calculate the covariance of Amazon and S&P500
covariance = np.cov(data['AMZN.O'], data['SPY'])[0][1]


# In[43]:


covariance


# In[44]:


# Calculate the variance of S&P500
variance = np.var(data['SPY'])


# In[45]:


variance


# In[46]:


# Print Beta of Amazon
print(f"Beta for Amazon {covariance/variance}")


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://github.com/Datatouille/findalpha/blob/master/slides.pdf
# - [Data](https://github.com/yhilpisch/py4fi2nd/tree/master/source)
# 
# </font>
# </div>

# # Requirements
# <hr style = "border:2px solid black" ></hr>

# In[37]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -iv')

