#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#What-is-correlation?" data-toc-modified-id="What-is-correlation?-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>What is correlation?</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Import-data" data-toc-modified-id="Import-data-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Import data</a></span></li><li><span><a href="#Compute-correlation" data-toc-modified-id="Compute-correlation-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Compute correlation</a></span></li><li><span><a href="#References" data-toc-modified-id="References-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
#
# **What?** Correlation
#
# </font>
# </div>

# # What is correlation?
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# - Correlation measures association, but doesn't show if x causes y or vice versa
# - Correlation is a statistic that measures the degree to which two variables move in relation to each other.
# - **How is it used in finance?** It is used to quantify the movement of a stock with that of a benchmark index, such as the S&P 500.
#
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[1]:


import pandas as pd
import pandas_datareader as pdr
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt


# # Import data
# <hr style = "border:2px solid black" ></hr>

# In[2]:


tickers = ["AAPL", "TWTR", "IBM", "MSFT"]
start = dt.datetime(2020, 1, 1)

data = pdr.get_data_yahoo(tickers, start)


# In[3]:


data.head()


# In[4]:


data = data["Adj Close"]


# In[5]:


data.head()


# <div class="alert alert-info">
# <font color=black>
#
# - Calculate the log returns.
# -  Remember we do it on the log returns to keep different stock index on the same range.
#
# </font>
# </div>

# In[6]:


log_returns = np.log(data / data.shift())


# In[7]:


log_returns


# In[8]:


log_returns.corr()


# <div class="alert alert-info">
# <font color=black>
#
# - Correlation on the diagonal is always1.0. This is obvious, since the diagonal shows the correlation between itself (AAPL and AAPL, and so forth).
#
# - Other than that, we can conclude that AAPL and MSFT are correlated the most.
#
# - Let’s add the S&P 500 to our DataFrame. This is done because the performance is generally benchmarked against this large agglomerate of best 500 performants.
#
# </font>
# </div>

# In[9]:


sp500 = pdr.get_data_yahoo("^GSPC", start)


# In[10]:


log_returns["SP500"] = np.log(sp500["Adj Close"] / sp500["Adj Close"].shift())


# In[11]:


log_returns.corr()


# <div class="alert alert-info">
# <font color=black>
#
# - AAPL and MSFT are mostly correlated to S&P 500 index.
# - This is not surprising, as they are a big part of the weight of the market cap in the index.
#
# </font>
# </div>

# # Compute correlation
# <hr style = "border:2px solid black" ></hr>

# In[12]:


def test_correlation(ticker):
    df = pdr.get_data_yahoo(ticker, start)
    lr = log_returns.copy()
    lr[ticker] = np.log(df["Adj Close"] / df["Adj Close"].shift())
    return lr.corr()


# In[13]:


def visualize_correlation(ticker1, ticker2):
    df = pdr.get_data_yahoo([ticker1, ticker2], start)
    df = df["Adj Close"]
    df = df / df.iloc[0]
    fig, ax = plt.subplots()
    df.plot(ax=ax)


# In[14]:


test_correlation("LQD")


# In[15]:


test_correlation("TLT")


# In[16]:


visualize_correlation("AAPL", "TLT")


# In[17]:


visualize_correlation("^GSPC", "TLT")


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
#
# - [GitHub code](https://github.com/LearnPythonWithRune/PythonForFinanceRiskAndReturn/blob/main/05%20-%20Correlation.ipynb)
# - [Blog article](https://www.learnpythonwithrune.org/python-for-finance-risk-and-return/#lesson-5)
# - Correlation https://www.investopedia.com/terms/c/correlation.asp
# - SP500 by Market Cap https://www.slickcharts.com/sp500
#
# </font>
# </div>

# In[ ]:
