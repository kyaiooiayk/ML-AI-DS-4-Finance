#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Definition" data-toc-modified-id="Definition-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Definition</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Import-data" data-toc-modified-id="Import-data-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Import data</a></span></li><li><span><a href="#Defining-a-portfoilio" data-toc-modified-id="Defining-a-portfoilio-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Defining a portfoilio</a></span></li><li><span><a href="#Capturing-trade-off-in-a-single-number" data-toc-modified-id="Capturing-trade-off-in-a-single-number-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Capturing trade-off in a single number</a></span></li><li><span><a href="#Sharpe-Ratio" data-toc-modified-id="Sharpe-Ratio-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Sharpe Ratio</a></span></li><li><span><a href="#References" data-toc-modified-id="References-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
#
# **What?** Sharpe Ratio
#
# </font>
# </div>

# # Definition
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# - Represents both the risk and return
# - Developed by Nobel laureate William F. Sharpe and is used to help investors understand the return of an investment compared to its ris
# - Goal is to get high $SR$.
#
# $SR = \frac{R_p - R_f}{\sigma_p}$
#
#
# - $SR$: Sharpe ratio
# - $R_p$: return of portfolio
# - $R_f$: risk free return
# - $\sigma_p$: standard deviation of portfolio
#
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[1]:


import numpy as np
import pandas_datareader as pdr
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt


# # Import data
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# - Where our portfolio will consist of the tickers for Apple, Microsoft, Twitter and IBM (AAPL, MSFT, TWTR, IBM). We read the data from start 2020 from the Yahoo! Finance API using Pandas Datareader.
#
# - Finally, we only keep the Adjusted Close price.
#
# </font>
# </div>

# In[2]:


tickers = ["AAPL", "MSFT", "TWTR", "IBM"]
start = dt.datetime(2020, 1, 1)

data = pdr.get_data_yahoo(tickers, start)


# In[3]:


data.head()


# In[4]:


data = data["Adj Close"]


# In[5]:


data


# #  Defining a portfoilio
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# - Let’s assume our portfolio is balanced as follows, 25%, 15%, 40%, and 20% to AAPL, MSFT, TWTR, IBM, respectively.
#
# - Then we can calculate the daily log return of the portfolio.
#
# - This gives an impression of how volatile the portfolio is. The more data is centered around 0.0, the less volatile and risky.
#
# </font>
# </div>

# In[6]:


portfolio = [0.25, 0.15, 0.40, 0.20]
log_return = np.sum(np.log(data / data.shift()) * portfolio, axis=1)


# In[7]:


log_return


# In[9]:


fig, ax = plt.subplots()
log_return.hist(bins=50, ax=ax)


# # Capturing trade-off in a single number
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# - The return and risk objectives imply a trade-off: taking more risk may yield higher returns in some circumstances, but also implies greater downside.
#
# - To compare how different strategies navigate this trade-off, ratios that compute a measure of return per unit of risk are very popular.
#
# - Two of the most popular are:
#     - Sharpe ratio (SR)
#     - Information ratio (IR)
#
# </font>
# </div>

# # Sharpe Ratio
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# - This gives a daily Sharpe Ratio, where we have the return to be the mean value. That is, the average return of the investment. And divided by the standard deviation.
#
# - The greater is the standard deviation the greater the magnitude of the deviation from the mean value can be expected.
#
# </font>
# </div>

# In[12]:


sharpe_ratio = log_return.mean() / log_return.std()
sharpe_ratio


# In[13]:


# To get an annualized Sharpe Ratio
sharpe_ratio_annual = sharpe_ratio * 252**0.5
sharpe_ratio_annual


# In[1]:


252**0.5


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
#
# - [GitHub code](https://github.com/LearnPythonWithRune/PythonForFinanceRiskAndReturn/blob/main/03%20-%20Sharpe%20Ratio.ipynb)
# - https://www.learnpythonwithrune.org/python-for-finance-risk-and-return/#lesson-3
# - https://www.investopedia.com/terms/s/sharperatio.asp
# - Jansen, Stefan. Hands-On Machine Learning for Algorithmic Trading: Design and implement investment strategies based on smart algorithms that learn from data using Python. Packt Publishing Ltd, 2018.
#
# </font>
# </div>

# In[ ]:
