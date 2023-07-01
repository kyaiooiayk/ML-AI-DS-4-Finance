#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Read-in-data" data-toc-modified-id="Read-in-data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Read-in data</a></span></li><li><span><a href="#Simple-EDA" data-toc-modified-id="Simple-EDA-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Simple EDA</a></span></li><li><span><a href="#Resampling" data-toc-modified-id="Resampling-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Resampling</a></span></li><li><span><a href="#References" data-toc-modified-id="References-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>References</a></span></li><li><span><a href="#Requirements" data-toc-modified-id="Requirements-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Requirements</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Resampling high-frequency data
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
get_ipython().run_line_magic('matplotlib', 'notebook')


# # Read-in data
# <hr style = "border:2px solid black" ></hr>

# In[2]:


get_ipython().run_cell_magic('time', '', "# data from FXCM Forex Capital Markets Ltd.\ntick = pd.read_csv('../data/fxcm_eur_usd_tick_data.csv',\n                     index_col=0, parse_dates=True)")


# In[3]:


tick.info()


# # Simple EDA
# <hr style = "border:2px solid black" ></hr>

# In[4]:


tick['Mid'] = tick.mean(axis=1)  


# In[5]:


tick['Mid'].plot(figsize=(10, 6))


# # Resampling
# <hr style = "border:2px solid black" ></hr>

# In[6]:


tick_resam = tick.resample(rule='5min', label='right').last()


# In[7]:


tick_resam.head()


# In[8]:


tick_resam['Mid'].plot(figsize=(10, 6));


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

# In[9]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -iv')


# In[ ]:




