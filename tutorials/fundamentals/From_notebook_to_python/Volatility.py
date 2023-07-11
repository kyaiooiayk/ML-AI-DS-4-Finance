#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Get-the-data" data-toc-modified-id="Get-the-data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Get the data</a></span></li><li><span><a href="#Compute-volatility" data-toc-modified-id="Compute-volatility-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Compute volatility</a></span></li><li><span><a href="#References" data-toc-modified-id="References-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
#
# **What?** Volatily
#
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[1]:


import pandas_datareader as pdr
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # Get the data
# <hr style = "border:2px solid black" ></hr>

# In[2]:


start = dt.datetime(1999, 1, 1)
end = dt.datetime(2008, 12, 31)

data = pdr.get_data_yahoo("^GSPC", start, end)


# In[3]:


data


# In[4]:


data["Log returns"] = np.log(data["Adj Close"] / data["Adj Close"].shift())


# In[5]:


data["Adj Close"].iloc[-1] / data["Adj Close"].iloc[0]


# In[6]:


data["Log returns"].sum()


# In[7]:


np.exp(data["Log returns"].sum())


# In[8]:


data["Normalize"] = data["Adj Close"] / data["Adj Close"].iloc[0]


# In[9]:


data["Exp sum"] = data["Log returns"].cumsum().apply(np.exp)


# In[10]:


data[["Normalize", "Exp sum"]].tail()


# # Compute volatility
# <hr style = "border:2px solid black" ></hr>

# ![image.png](attachment:image.png)

# In[11]:


volatility = data["Log returns"].std() * (252**0.5)


# In[12]:


volatility


# In[13]:


str_vol = str(round(volatility, 3) * 100)


# In[14]:


fig, ax = plt.subplots()
data["Log returns"].hist(ax=ax, bins=50, alpha=0.6, color="b")
ax.set_xlabel("Log returns of stock price")
ax.set_ylabel("Frequencey of log returns")
ax.set_title("Historic Volatility for S&P 500 (" + str_vol + "%)")


# In[15]:


np.log(1.2) + np.log(1.15) + np.log(1.1) + np.log(1.3)


# In[16]:


np.exp(0.6797579434409292)


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
#
# - https://github.com/LearnPythonWithRune/PythonForFinancialAnalysis/blob/main/04%20-%20Volatility.ipynb
# - https://www.learnpythonwithrune.org/start-python-with-pandas-for-financial-analysis/#lesson-6
#
# </font>
# </div>

# In[ ]:
