#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#What-is-CAPM?" data-toc-modified-id="What-is-CAPM?-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>What is CAPM?</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Load-data" data-toc-modified-id="Load-data-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Load data</a></span></li><li><span><a href="#CAPM" data-toc-modified-id="CAPM-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>CAPM</a></span></li><li><span><a href="#References" data-toc-modified-id="References-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Capital Asset Pricing Model (CAPM)
# 
# </font>
# </div>

# # What is CAPM?
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - Relationship between systematic risk and expected return
# - There are several assumptions behind the CAPM formula that have been shown not to hold in reality.
# - CAPM formula is still widely used:
# 
#  $ER_i = R_f + \beta_i(ER_m - R_f)$
# 
# 
# * $ER_i$: Expected return from investment
# * $R_f$: Risk free return
# * $\beta_i$: The beta of the investment
# * $(ER_m - R_f)$: Market risk premium   
# </font>
# </div>
# 
#     

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[1]:


import numpy as np
import pandas_datareader as pdr
import datetime as dt
import pandas as pd


# # Load data
# <hr style = "border:2px solid black" ></hr>

# In[2]:


tickers = ['AAPL', 'MSFT', 'TWTR', 'IBM', '^GSPC']
start = dt.datetime(2015, 12, 1)
end = dt.datetime(2021, 1, 1)

data = pdr.get_data_yahoo(tickers, start, end, interval="m")


# In[3]:


data = data['Adj Close']


# In[4]:


log_returns = np.log(data/data.shift())


# In[5]:


cov = log_returns.cov()
var = log_returns['^GSPC'].var()


# In[6]:


beta = cov.loc['AAPL', '^GSPC']/var


# # CAPM
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - The risk free return is often set to 0. Otherwise, the [10 years treasury note is used](https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/textview.aspx?data=yield). Here, we use 1.38%. You can update it for more up to date value with the link.
# 
# </font>
# </div>

# In[7]:


risk_free_return = 0.0138
market_return = .105
expected_return = risk_free_return + beta*(market_return - risk_free_return)


# In[8]:


expected_return


# In[9]:


beta*market_return


# In[10]:


beta = cov.loc['^GSPC']/var


# In[11]:


beta


# In[12]:


market_return = risk_free_return + beta*(market_return - risk_free_return)


# In[13]:


market_return


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://github.com/LearnPythonWithRune/PythonForFinanceRiskAndReturn/blob/main/08%20-%20CAPM.ipynb
# - [Market risk premium](https://www.investopedia.com/terms/m/marketriskpremium.asp)
#     
# </font>
# </div>

# In[ ]:




