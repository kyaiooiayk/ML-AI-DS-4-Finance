#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Select-ticker" data-toc-modified-id="Select-ticker-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Select ticker</a></span></li><li><span><a href="#References" data-toc-modified-id="References-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** YAHOO finacial data API
# 
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[7]:


from pandas_datareader import data as pdr
import datetime as dt
import yfinance as yfin
yfin.pdr_override()


# # Select ticker
# <hr style = "border:2px solid black" ></hr>

# In[8]:


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


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://github.com/LearnPythonWithRune/PythonForFinancialAnalysis/blob/main/01%20-%20Read%20from%20API.ipynb
# - https://www.learnpythonwithrune.org/start-python-with-pandas-for-financial-analysis/
# - https://pythonprogramming.net/getting-stock-prices-python-programming-for-finance/
# - https://www.quantstart.com/articles/understanding-equities-data/
# - [typeerror: string indices must be integer pandas datareader](https://stackoverflow.com/questions/74912452/typeerror-string-indices-must-be-integer-pandas-datareader)
#     
# </font>
# </div>

# In[ ]:




