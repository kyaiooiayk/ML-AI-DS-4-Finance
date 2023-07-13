#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Load-data" data-toc-modified-id="Load-data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Load data</a></span></li><li><span><a href="#Select-stocks" data-toc-modified-id="Select-stocks-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Select stocks</a></span></li><li><span><a href="#Normalise-data" data-toc-modified-id="Normalise-data-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Normalise data</a></span></li><li><span><a href="#Chose-portfolio-weights" data-toc-modified-id="Chose-portfolio-weights-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Chose portfolio weights</a></span></li><li><span><a href="#Get-the-returns" data-toc-modified-id="Get-the-returns-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Get the returns</a></span></li><li><span><a href="#References" data-toc-modified-id="References-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** How to construct a very simple portfolio
# 
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[1]:


import pandas_datareader as pdr
import datetime as dt


# # Load data
# <hr style = "border:2px solid black" ></hr>

# In[2]:


start = dt.datetime(2010, 1, 1)

aapl = pdr.get_data_yahoo("AAPL", start)


# <div class="alert alert-info">
# <font color=black>
# 
# - `Open` - When the stock market opens in the morning for trading, what was the price of one share?
# - `High` - over the course of the trading day, what was the highest value for that day?
# - `Low` - over the course of the trading day, what was the lowest value for that day?
# - `Close` - When the trading day was over, what was the final price?
# - `Volume` - For that day, how many shares were traded?
# - `Adj Close` - Over time, companies may decide to do something called a stock split. For example, Apple did one once their stock price exceeded 1000USD. Since in most cases, people cannot buy fractions of shares, a stock price of 1,000 USD is fairly limiting to investors. Companies can do a stock split where they say every share is now 2 shares, and the price is half. Anyone who had 1 share of Apple for 1,000 USD, after a split where Apple doubled the shares, they would have 2 shares of Apple (AAPL), each worth 500 USD. Adj Close is helpful, since it accounts for future stock splits, and gives the relative price to splits. For this reason, the **adjusted** prices are the prices you're most likely to be dealing with.
# 
# </font>
# </div>

# In[3]:


aapl.head()


# In[4]:


start = dt.datetime(2010, 1, 1)
end = dt.datetime(2020, 1, 1)

aapl = pdr.get_data_yahoo("AAPL", start, end)


# In[5]:


aapl


# # Select stocks
# <hr style = "border:2px solid black" ></hr>

# In[6]:


tickers = ['AAPL', 'MSFT', 'NFLX', 'AMZN']

start = dt.datetime(2010, 1, 1)

data = pdr.get_data_yahoo(tickers, start)


# In[7]:


data.head()


# # Normalise data
# <hr style = "border:2px solid black" ></hr>

# In[8]:


data = data['Adj Close']


# In[9]:


data.head()


# In[10]:


data.iloc[0]


# In[11]:


# normalisation wrt the first entry
norm = data/data.iloc[0]


# In[12]:


norm.head()


# # Chose portfolio weights
# <hr style = "border:2px solid black" ></hr>

# In[13]:


portfolio = [.25, .25, .25, .25]


# In[21]:


weights = norm*portfolio


# In[22]:


weights.head()


# In[16]:


# Calculate the sum and add index
weights['Total'] = (norm*portfolio).sum(axis=1)


# In[17]:


weights.head()


# # Get the returns
# <hr style = "border:2px solid black" ></hr>

# In[18]:


(weights*100000).head()


# In[19]:


(weights*100000).tail()


# In[20]:


(weights['Total']*100000).iloc[-1]


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://github.com/LearnPythonWithRune/PythonForFinancialAnalysis/blob/main/01%20-%20Read%20from%20API.ipynb
# - https://www.learnpythonwithrune.org/start-python-with-pandas-for-financial-analysis/
# - https://pythonprogramming.net/getting-stock-prices-python-programming-for-finance/
# 
# </font>
# </div>
