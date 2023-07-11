#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Binomial-Trees" data-toc-modified-id="Binomial-Trees-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Binomial Trees</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Python" data-toc-modified-id="Python-3.0.1"><span class="toc-item-num">3.0.1&nbsp;&nbsp;</span>Python</a></span></li><li><span><a href="#NumPy" data-toc-modified-id="NumPy-3.0.2"><span class="toc-item-num">3.0.2&nbsp;&nbsp;</span>NumPy</a></span></li><li><span><a href="#Numba" data-toc-modified-id="Numba-3.0.3"><span class="toc-item-num">3.0.3&nbsp;&nbsp;</span>Numba</a></span></li><li><span><a href="#Cython" data-toc-modified-id="Cython-3.0.4"><span class="toc-item-num">3.0.4&nbsp;&nbsp;</span>Cython</a></span></li></ul></li></ul></li><li><span><a href="#References" data-toc-modified-id="References-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
#
# **What?** Binomial option pricing model
#
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[1]:


import random
import numpy as np
import numba


# # Binomial Trees
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# - A popular numerical method to value options is the binomial option pricing model pioneered by Cox, Ross, and Rubinstein (1979).
# - This method relies on representing the possible future evolution of an asset by a (recombining) tree.
# - In this model, as in the Black-Scholes-Merton (1973) setup, there is a risky asset, an index or stock, and a riskless asset, a bond.
#
# </font>
# </div>

# ### Python

# In[2]:


import math


# In[3]:


S0 = 36.0
T = 1.0
r = 0.06
sigma = 0.2


# In[4]:


def simulate_tree(M):
    dt = T / M
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    S = np.zeros((M + 1, M + 1))
    S[0, 0] = S0
    z = 1
    for t in range(1, M + 1):
        for i in range(z):
            S[i, t] = S[i, t - 1] * u
            S[i + 1, t] = S[i, t - 1] * d
        z += 1
    return S


# In[5]:


np.set_printoptions(formatter={"float": lambda x: "%6.2f" % x})


# In[6]:


simulate_tree(4)


# In[7]:


get_ipython().run_line_magic("time", "simulate_tree(500)")


# ### NumPy

# In[8]:


M = 4


# In[9]:


up = np.arange(M + 1)
up = np.resize(up, (M + 1, M + 1))
up


# In[10]:


down = up.T * 2
down


# In[11]:


up - down


# In[12]:


dt = T / M


# In[13]:


S0 * np.exp(sigma * math.sqrt(dt) * (up - down))


# In[14]:


def simulate_tree_np(M):
    dt = T / M
    up = np.arange(M + 1)
    up = np.resize(up, (M + 1, M + 1))
    down = up.transpose() * 2
    S = S0 * np.exp(sigma * math.sqrt(dt) * (up - down))
    return S


# In[15]:


simulate_tree_np(4)


# In[16]:


get_ipython().run_line_magic("time", "simulate_tree_np(500)")


# ### Numba

# In[17]:


simulate_tree_nb = numba.jit(simulate_tree)


# In[18]:


simulate_tree_nb(4)


# In[19]:


get_ipython().run_line_magic("time", "simulate_tree_nb(500)")


# In[20]:


get_ipython().run_line_magic("timeit", "simulate_tree_nb(500)")


# ### Cython

# In[21]:


get_ipython().run_line_magic("load_ext", "Cython")


# In[22]:


get_ipython().run_cell_magic(
    "cython",
    "-a",
    "import numpy as np\ncimport cython\nfrom libc.math cimport exp, sqrt\ncdef float S0 = 36.\ncdef float T = 1.0\ncdef float r = 0.06\ncdef float sigma = 0.2\ndef simulate_tree_cy(int M):\n    cdef int z, t, i\n    cdef float dt, u, d\n    cdef float[:, :] S = np.zeros((M + 1, M + 1),\n                                  dtype=np.float32)  \n    dt = T / M\n    u = exp(sigma * sqrt(dt))\n    d = 1 / u\n    S[0, 0] = S0\n    z = 1\n    for t in range(1, M + 1):\n        for i in range(z):\n            S[i, t] = S[i, t-1] * u\n            S[i+1, t] = S[i, t-1] * d\n        z += 1\n    return np.array(S)\n",
)


# In[23]:


simulate_tree_cy(4)


# In[24]:


get_ipython().run_line_magic("time", "simulate_tree_cy(500)")


# In[25]:


get_ipython().run_line_magic("timeit", "S = simulate_tree_cy(500)")


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
