#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Exponentially-weighted-moving-average---EWMA" data-toc-modified-id="Exponentially-weighted-moving-average---EWMA-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Exponentially weighted moving average - EWMA</a></span></li><li><span><a href="#Import-data" data-toc-modified-id="Import-data-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Import data</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Pure-Python" data-toc-modified-id="Pure-Python-4.0.1"><span class="toc-item-num">4.0.1&nbsp;&nbsp;</span>Pure Python</a></span></li><li><span><a href="#Numba" data-toc-modified-id="Numba-4.0.2"><span class="toc-item-num">4.0.2&nbsp;&nbsp;</span>Numba</a></span></li><li><span><a href="#Cython" data-toc-modified-id="Cython-4.0.3"><span class="toc-item-num">4.0.3&nbsp;&nbsp;</span>Cython</a></span></li></ul></li></ul></li><li><span><a href="#References" data-toc-modified-id="References-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Exponentially weighted moving average - EWMA
# 
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[1]:


import random
import numpy as np
import numba
import pandas as pd


# # Exponentially weighted moving average - EWMA
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - While pandas allows for sophisticated vectorized operations on DataFrame objects, certain recursive algorithms are hard or impossible to vectorize, leaving the financial analyst with slowly executed Python loops on DataFrame objects.
# - Although simple in nature and straightforward to implement, such an algorithm might lead to rather slow code.
# 
# </font>
# </div>

# ![image.png](attachment:image.png)

# # Import data
# <hr style = "border:2px solid black" ></hr>

# In[2]:


sym = 'SPY'


# In[3]:


data = pd.DataFrame(pd.read_csv('http://hilpisch.com/tr_eikon_eod_data.csv',
                               index_col=0, parse_dates=True)[sym]).dropna()


# In[4]:


alpha = 0.25


# In[5]:


data['EWMA'] = data[sym]  


# In[6]:


get_ipython().run_cell_magic('time', '', "for t in zip(data.index, data.index[1:]):\n    data.loc[t[1], 'EWMA'] = (alpha * data.loc[t[1], sym] +\n                              (1 - alpha) * data.loc[t[0], 'EWMA'])  \n")


# In[7]:


data.head()


# In[8]:


data.shape


# In[9]:


data[data.index > '2017-1-1'].plot(figsize=(10, 6));


# ### Pure Python

# In[10]:


def ewma_py(x, alpha):
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1-alpha) * y[i-1]
    return y


# In[11]:


get_ipython().run_line_magic('time', "data['EWMA_PY'] = ewma_py(data[sym], alpha)")


# In[12]:


get_ipython().run_line_magic('time', "data['EWMA_PY'] = ewma_py(data[sym].values, alpha)")


# ### Numba

# In[13]:


ewma_nb = numba.jit(ewma_py)


# In[14]:


get_ipython().run_line_magic('time', "data['EWMA_NB'] = ewma_nb(data[sym].values, alpha)")


# In[15]:


get_ipython().run_line_magic('timeit', "data['EWMA_NB'] = ewma_nb(data[sym].values, alpha)")


# ### Cython

# In[24]:


get_ipython().run_line_magic('load_ext', 'Cython')


# In[25]:


get_ipython().run_cell_magic('cython', '', 'import numpy as np\ncimport cython\n@cython.boundscheck(False)\n@cython.wraparound(False)\ndef ewma_cy(double[:] x, float alpha):\n    cdef int i\n    cdef double[:] y = np.empty_like(x)\n    y[0] = x[0]\n    for i in range(1, len(x)):\n        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]\n    return y\n')


# In[26]:


get_ipython().run_line_magic('time', "data['EWMA_CY'] = ewma_cy(data[sym].values, alpha)")


# In[27]:


get_ipython().run_line_magic('timeit', "data['EWMA_CY'] = ewma_cy(data[sym].values, alpha)")


# In[28]:


data.head()


# In[29]:


data.tail()


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://github.com/yhilpisch/py4fi2nd/blob/master/code/ch10/10_performance_python.ipynb
# - https://llvm.org/
# - Hilpisch, Yves. Python for finance: mastering data-driven finance. O'Reilly Media, 2018.
# 
# </font>
# </div>

# In[ ]:




