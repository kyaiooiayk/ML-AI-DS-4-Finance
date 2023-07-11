#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Monte-Carlo-simulation" data-toc-modified-id="Monte-Carlo-simulation-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Monte Carlo simulation</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Python" data-toc-modified-id="Python-3.0.1"><span class="toc-item-num">3.0.1&nbsp;&nbsp;</span>Python</a></span></li><li><span><a href="#NumPy" data-toc-modified-id="NumPy-3.0.2"><span class="toc-item-num">3.0.2&nbsp;&nbsp;</span>NumPy</a></span></li><li><span><a href="#Numba" data-toc-modified-id="Numba-3.0.3"><span class="toc-item-num">3.0.3&nbsp;&nbsp;</span>Numba</a></span></li><li><span><a href="#Cython-—-Sequential" data-toc-modified-id="Cython-—-Sequential-3.0.4"><span class="toc-item-num">3.0.4&nbsp;&nbsp;</span>Cython — Sequential</a></span></li><li><span><a href="#Multiprocessing" data-toc-modified-id="Multiprocessing-3.0.5"><span class="toc-item-num">3.0.5&nbsp;&nbsp;</span>Multiprocessing</a></span></li></ul></li></ul></li><li><span><a href="#References" data-toc-modified-id="References-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>References</a></span></li><li><span><a href="#Requirements" data-toc-modified-id="Requirements-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Requirements</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
#
# **What?** Monte Carlo simulation of the geometric Brownian motion
#
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[1]:


import random
import numpy as np
import numba
import math
import matplotlib.pyplot as plt
import multiprocessing as mp


# # Monte Carlo simulation
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# - This section analyzes the **Monte Carlo** simulation of the geometric Brownian motion, a simple yet still widely used stochastic process to model the evolution of stock prices or index levels.
#
# - MCS is among the most important numerical techniques in finance, if not the most important and widely used. This mainly stems from the fact that it is the most flexible numerical method when it comes to the evaluation of mathematical expressions (e.g., integrals), and specifically the valuation of financial derivatives. The flexibility comes at the cost of a relatively high computational burden, though, since often hundreds of thousands or even millions of complex computations have to be carried out to come up with a single value estimate.
#
# - Among others, the **Black-Scholes-Merton** (1973) theory of option pricing draws on this process. In their setup the underlying of the option to be valued follows the stochastic differential equation (SDE),
#
# </font>
# </div>

# ### Python

# In[2]:


# Initial value of the risky asset.
S0 = 36.0
# Time horizon for the binomial tree simulation.
T = 1.0
# Constant short rate.
r = 0.06
# Constant volatility factor.
sigma = 0.2


# In[3]:


# The number of time intervals for discretization
M = 100
# The number of paths to be simulated.
I = 50000


# In[4]:


def mcs_simulation_py(p):
    M, I = p
    T = 1
    # Length of the time intervals.
    dt = T / M
    S = np.zeros((M + 1, I))
    S[0] = S0
    rn = np.random.standard_normal(S.shape)
    for t in range(1, M + 1):
        for i in range(I):
            S[t, i] = S[t - 1, i] * math.exp(
                (r - sigma**2 / 2) * dt + sigma * math.sqrt(dt) * rn[t, i]
            )
    return S


# In[5]:


get_ipython().run_line_magic("time", "S = mcs_simulation_py((M, I))")


# In[6]:


S[-1].mean()


# In[7]:


S0 * math.exp(r * T)


# In[8]:


K = 40.0


# In[9]:


C0 = math.exp(-r * T) * np.maximum(K - S[-1], 0).mean()


# In[10]:


C0


# In[11]:


plt.figure(figsize=(10, 6))
plt.hist(S[-1], bins=35, label="frequency")
plt.axvline(S[-1].mean(), color="r", label="mean value")
plt.legend(loc=0)


# ### NumPy

# In[12]:


def mcs_simulation_np(p):
    M, I = p
    dt = T / M
    S = np.zeros((M + 1, I))
    S[0] = S0
    rn = np.random.standard_normal(S.shape)
    for t in range(1, M + 1):
        S[t] = S[t - 1] * np.exp(
            (r - sigma**2 / 2) * dt + sigma * math.sqrt(dt) * rn[t]
        )
    return S


# In[13]:


get_ipython().run_line_magic("time", "S = mcs_simulation_np((M, I))")


# In[14]:


S[-1].mean()


# In[15]:


get_ipython().run_line_magic("timeit", "S = mcs_simulation_np((M, I))")


# ### Numba

# In[16]:


mcs_simulation_nb = numba.jit(mcs_simulation_py)


# In[17]:


get_ipython().run_line_magic("time", "S = mcs_simulation_nb((M, I))")


# In[18]:


get_ipython().run_line_magic("time", "S = mcs_simulation_nb((M, I))")


# In[19]:


S[-1].mean()


# In[20]:


C0 = math.exp(-r * T) * np.maximum(K - S[-1], 0).mean()


# In[21]:


C0


# In[22]:


get_ipython().run_line_magic("timeit", "S = mcs_simulation_nb((M, I))")


# ### Cython &mdash; Sequential

# In[23]:


get_ipython().run_line_magic("load_ext", "Cython")


# In[24]:


get_ipython().run_cell_magic(
    "cython",
    "",
    "import numpy as np\ncimport numpy as np\ncimport cython\nfrom libc.math cimport exp, sqrt\ncdef float S0 = 36.\ncdef float T = 1.0\ncdef float r = 0.06\ncdef float sigma = 0.2\n@cython.boundscheck(False)\n@cython.wraparound(False)\ndef mcs_simulation_cy(p):\n    cdef int M, I\n    M, I = p\n    cdef int t, i\n    cdef float dt = T / M\n    cdef double[:, :] S = np.zeros((M + 1, I))\n    cdef double[:, :] rn = np.random.standard_normal((M + 1, I))\n    S[0] = S0\n    for t in range(1, M + 1):\n        for i in range(I):\n            S[t, i] = S[t-1, i] * exp((r - sigma ** 2 / 2) * dt +\n                                         sigma * sqrt(dt) * rn[t, i])\n    return np.array(S) \n",
)


# In[25]:


get_ipython().run_line_magic("time", "S = mcs_simulation_cy((M, I))")


# In[26]:


S[-1].mean()


# In[27]:


get_ipython().run_line_magic("timeit", "S = mcs_simulation_cy((M, I))")


# ### Multiprocessing

# In[28]:


pool = mp.Pool(processes=-1)


# In[29]:


p = 20


# In[30]:


get_ipython().run_line_magic(
    "timeit", "S = np.hstack(pool.map(mcs_simulation_np, p * [(M, int(I / p))]))"
)


# In[ ]:


get_ipython().run_line_magic(
    "timeit", "S = np.hstack(pool.map(mcs_simulation_nb, p * [(M, int(I / p))]))"
)


# In[ ]:


get_ipython().run_line_magic(
    "timeit", "S = np.hstack(pool.map(mcs_simulation_cy, p * [(M, int(I / p))]))"
)


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
#
# - https://github.com/yhilpisch/py4fi2nd/blob/master/code/ch10/10_performance_python.ipynb
# - https://llvm.org/
# - https://github.com/yhilpisch/py4fi2nd/blob/master/code/ch12/12_stochastics.ipynb
# - Hilpisch, Yves. Python for finance: mastering data-driven finance. O'Reilly Media, 2018.
#
# </font>
# </div>

# # Requirements
# <hr style = "border:2px solid black" ></hr>

# In[1]:


get_ipython().run_line_magic("load_ext", "watermark")
get_ipython().run_line_magic("watermark", "-v -iv")


# In[ ]:
