#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#References" data-toc-modified-id="References-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
#
# **What?** EDA for financial data - candlesticks
#
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[2]:


start = dt(2016, 11, 1)
end = dt(2021, 11, 1)
aapl = web.DataReader("AAPl", "yahoo", start, end)


# In[3]:


aapl


# In[4]:


aapl.plot(y="Adj Close")


# In[5]:


start = dt(2021, 10, 1)
end = dt(2021, 11, 1)
goog = web.DataReader("GOOG", "yahoo", start, end)


# In[6]:


# define the data
candlestick = go.Candlestick(
    x=aapl.index,
    open=goog["Open"],
    high=goog["High"],
    low=goog["Low"],
    close=goog["Close"],
    name="OHLC",
)

# create the figure
fig = go.Figure(data=[candlestick])

# plot the figure
fig.show()


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
#
# - https://www.quantstart.com/articles/creating-an-algorithmic-trading-prototyping-environment-with-jupyter-notebooks-and-plotly/
#
# </font>
# </div>

# In[ ]:
