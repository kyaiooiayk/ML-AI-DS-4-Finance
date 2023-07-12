#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Portfolio-Optimization-using-Second-Order-Cone" data-toc-modified-id="Portfolio-Optimization-using-Second-Order-Cone-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Portfolio Optimization using Second Order Cone</a></span><ul class="toc-item"><li><span><a href="#1.-Variance-Optimization" data-toc-modified-id="1.-Variance-Optimization-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>1. Variance Optimization</a></span><ul class="toc-item"><li><span><a href="#1.1-Variance-Minimization" data-toc-modified-id="1.1-Variance-Minimization-2.1.1"><span class="toc-item-num">2.1.1&nbsp;&nbsp;</span>1.1 Variance Minimization</a></span></li><li><span><a href="#1.2-Return-Maximization-with-Variance-Constraint" data-toc-modified-id="1.2-Return-Maximization-with-Variance-Constraint-2.1.2"><span class="toc-item-num">2.1.2&nbsp;&nbsp;</span>1.2 Return Maximization with Variance Constraint</a></span></li></ul></li><li><span><a href="#2-Standard-Deviation-Optimization" data-toc-modified-id="2-Standard-Deviation-Optimization-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>2 Standard Deviation Optimization</a></span><ul class="toc-item"><li><span><a href="#2.1-Standard-Deviation-Minimization" data-toc-modified-id="2.1-Standard-Deviation-Minimization-2.2.1"><span class="toc-item-num">2.2.1&nbsp;&nbsp;</span>2.1 Standard Deviation Minimization</a></span></li><li><span><a href="#2.2-Return-Maximization-with-Standard-Deviation-Constraint" data-toc-modified-id="2.2-Return-Maximization-with-Standard-Deviation-Constraint-2.2.2"><span class="toc-item-num">2.2.2&nbsp;&nbsp;</span>2.2 Return Maximization with Standard Deviation Constraint</a></span></li></ul></li></ul></li><li><span><a href="#References" data-toc-modified-id="References-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# # Portfolio Optimization using Second Order Cone
#
# In this notebook we show how to use the Second Order Cone (SOC) constraint in the variance portfolio optimization problem.
#
# ## 1. Variance Optimization
#
# ### 1.1 Variance Minimization
#
# The minimization of portfolio variance is a quadratic optimization problem that can be posed as:
#
# $$
# \begin{equation}
# \begin{aligned}
# & \underset{x}{\text{min}} & &  x^{\tau} \Sigma x \\
# & \text{s.t.} & & \mu x^{\tau} \geq \bar{\mu} \\
# & & &  \sum_{i=1}^{N} x_i = 1 \\
# & & &  x_i \geq 0 \; ; \; \forall \; i =1, \ldots, N \\
# \end{aligned}
# \end{equation}
# $$
#
# Where $x$ are the weights of assets, $\mu$ is the mean vector of expected returns and $\bar{\mu}$ the minimum expected return of portfolio.

# In[1]:


####################################
# Downloading Data
####################################
get_ipython().system("pip install --quiet yfinance")

import numpy as np
import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

yf.pdr_override()
pd.options.display.float_format = "{:.4%}".format

# Date range
start = "2016-01-01"
end = "2019-12-30"

# Tickers of assets
assets = [
    "JCI",
    "TGT",
    "CMCSA",
    "CPB",
    "MO",
    "APA",
    "MMC",
    "JPM",
    "ZION",
    "PSA",
    "BAX",
    "BMY",
    "LUV",
    "PCAR",
    "TXT",
    "TMO",
    "DE",
    "MSFT",
    "HPQ",
    "SEE",
    "VZ",
    "CNP",
    "NI",
    "T",
    "BA",
]
assets.sort()

# Downloading data
data = yf.download(assets, start=start, end=end)
data = data.loc[:, ("Adj Close", slice(None))]
data.columns = assets

# Calculating returns
Y = data[assets].pct_change().dropna()

display(Y.head())


# In[2]:


####################################
# Minimizing Portfolio Variance
####################################

import cvxpy as cp
from timeit import default_timer as timer

# Defining initial inputs
mu = Y.mean().to_numpy().reshape(1, -1)
sigma = Y.cov().to_numpy()

# Defining initial variables
x = cp.Variable((mu.shape[1], 1))

# Budget and weights constraints
constraints = [cp.sum(x) == 1, x <= 1, x >= 0]

# Defining risk objective
risk = cp.quad_form(x, sigma)
objective = cp.Minimize(risk)

weights = pd.DataFrame([])
# Solving the problem with several solvers
prob = cp.Problem(objective, constraints)
solvers = ["ECOS", "SCS"]
for i in solvers:
    prob.solve(solver=i)
    # Showing Optimal Weights
    weights_1 = pd.DataFrame(x.value, index=assets, columns=[i])
    weights = pd.concat([weights, weights_1], axis=1)

display(weights)


# As we can see the use of CVXPY's __quad_form__ in portfolio optimization can give small negative values to weights that must be zero.

# In[3]:


# Calculating Annualized Portfolio Stats
var = weights * (Y.cov() @ weights) * 252
var = var.sum().to_frame().T
std = np.sqrt(var)
ret = Y.mean().to_frame().T @ weights * 252

stats = pd.concat([ret, std, var], axis=0)
stats.index = ["Return", "Std. Dev.", "Variance"]

display(stats)


# ### 1.2 Return Maximization with Variance Constraint
#
# The maximization of portfolio return is a problem with a quadratic constraint that can be posed as:
#
# $$
# \begin{equation}
# \begin{aligned}
# & \underset{x}{\text{max}} & & \mu x^{\tau} \\
# & \text{s.t.} & & x^{\tau} \Sigma x \leq \bar{\sigma}^{2} \\
# & & & \sum_{i=1}^{N} x_i = 1 \\
# & & &  x_i \geq 0 \; ; \; \forall \; i =1, \ldots, N \\
# \end{aligned}
# \end{equation}
# $$
#
# Where $x$ are the weights of assets and $\bar{\sigma}$ is the maximum expected standard deviation of portfolio..

# In[4]:


#########################################
# Maximizing Portfolio Return with
# Variance Constraint
#########################################

import cvxpy as cp
from timeit import default_timer as timer

# Defining initial inputs
mu = Y.mean().to_numpy().reshape(1, -1)
sigma = Y.cov().to_numpy()

# Defining initial variables
x = cp.Variable((mu.shape[1], 1))
sigma_hat = 15 / (252**0.5 * 100)
ret = mu @ x

# Budget and weights constraints
constraints = [cp.sum(x) == 1, x <= 1, x >= 0]

# Defining risk constraint and objective
risk = cp.quad_form(x, sigma)
constraints += [risk <= sigma_hat**2]  # variance constraint
objective = cp.Maximize(ret)

weights = pd.DataFrame([])
# Solving the problem with several solvers
prob = cp.Problem(objective, constraints)
solvers = ["ECOS", "SCS"]
for i in solvers:
    prob.solve(solver=i)
    # Showing Optimal Weights
    weights_1 = pd.DataFrame(x.value, index=assets, columns=[i])
    weights = pd.concat([weights, weights_1], axis=1)

display(weights)


# The small negative values also appear when we use CVXPY's __quad_form__ in constraints.

# In[5]:


# Calculating Annualized Portfolio Stats
var = weights * (Y.cov() @ weights) * 252
var = var.sum().to_frame().T
std = np.sqrt(var)
ret = Y.mean().to_frame().T @ weights * 252

stats = pd.concat([ret, std, var], axis=0)
stats.index = ["Return", "Std. Dev.", "Variance"]

display(stats)


# ## 2 Standard Deviation Optimization
#
# ### 2.1 Standard Deviation Minimization
#
# An alternative problem is to minimize the standard deviation (square root of variance). To do this we can use the SOC constraint. The minimization of portfolio standard deviation can be posed as:
#
# $$
# \begin{equation}
# \begin{aligned}
# & \underset{x}{\text{min}} & &  g \\
# & \text{s.t.} & & \mu x^{\tau} \geq \bar{\mu} \\
# & & &  \sum_{i=1}^{N} x_i = 1 \\
# & & & \left\|\Sigma^{1/2} x\right\| \leq g \\
# & & &  x_i \geq 0 \; ; \; \forall \; i =1, \ldots, N  \\
# \end{aligned}
# \end{equation}
# $$
#
# Where $\left\|\Sigma^{1/2} x\right\| \leq g$ is the SOC constraint, $x$ are the weights of assets, $\mu$ is the mean vector of expected returns, $\bar{\mu}$ the minimum expected return of portfolio and $r$ is the matrix of observed returns.
#
# __Note:__ the SOC constraint can be expressed as $(g,\Sigma^{1/2} x) \in Q^{n+1}$, this notation is used to model the __SOC constraint__ in CVXPY.

# In[6]:


#########################################
# Minimizing Portfolio Standard Deviation
#########################################

from scipy.linalg import sqrtm

# Defining initial inputs
mu = Y.mean().to_numpy().reshape(1, -1)
sigma = Y.cov().to_numpy()
G = sqrtm(sigma)

# Defining initial variables
x = cp.Variable((mu.shape[1], 1))
g = cp.Variable(nonneg=True)

# Budget and weights constraints
constraints = [cp.sum(x) == 1, x >= 0]

# Defining risk objective
risk = g
constraints += [cp.SOC(g, G @ x)]  # SOC constraint
constraints += [risk <= sigma_hat]  # variance constraint
objective = cp.Minimize(risk)

weights = pd.DataFrame([])
# Solving the problem with several solvers
prob = cp.Problem(objective, constraints)
solvers = ["ECOS", "SCS"]
for i in solvers:
    prob.solve(solver=i)
    # Showing Optimal Weights
    weights_1 = pd.DataFrame(x.value, index=assets, columns=[i])
    weights = pd.concat([weights, weights_1], axis=1)

display(weights)


# As we can see the use of CVXPY's __SOC constraint__ in portfolio optimization solves the error that we see when we use __quad_form__.

# In[7]:


# Calculating Annualized Portfolio Stats
var = weights * (Y.cov() @ weights) * 252
var = var.sum().to_frame().T
std = np.sqrt(var)
ret = Y.mean().to_frame().T @ weights * 252

stats = pd.concat([ret, std, var], axis=0)
stats.index = ["Return", "Std. Dev.", "Variance"]

display(stats)


# ### 2.2 Return Maximization with Standard Deviation Constraint
#
# The maximization of portfolio return using SOC constraints can be posed as:
#
# $$
# \begin{equation}
# \begin{aligned}
# & \underset{x}{\text{max}} & & \mu x^{\tau} \\
# & \text{s.t.} & & g \leq \bar{\sigma} \\
# & & & \left\|\Sigma^{1/2} x\right\| \leq g \\
# & & & \sum_{i=1}^{N} x_i = 1 \\
# & & &  x_i \geq 0 \; ; \; \forall \; i =1, \ldots, N \\
# \end{aligned}
# \end{equation}
# $$

# In[8]:


#########################################
# Maximizing Portfolio Return with
# Standard Deviation Constraint
#########################################

from scipy.linalg import sqrtm

# Defining initial inputs
mu = Y.mean().to_numpy().reshape(1, -1)
sigma = Y.cov().to_numpy()
G = sqrtm(sigma)

# Defining initial variables
x = cp.Variable((mu.shape[1], 1))
g = cp.Variable(nonneg=True)
sigma_hat = 15 / (252**0.5 * 100)
ret = mu @ x

# Budget and weights constraints
constraints = [cp.sum(x) == 1, x <= 1, x >= 0]


# Defining risk constraint and objective
risk = g
constraints += [cp.SOC(g, G @ x)]  # SOC constraint
constraints += [risk <= sigma_hat]  # standard deviation constraint
objective = cp.Maximize(ret)

weights = pd.DataFrame([])
# Solving the problem with several solvers
prob = cp.Problem(objective, constraints)
solvers = ["ECOS", "SCS"]
for i in solvers:
    prob.solve(solver=i)
    # Showing Optimal Weights
    weights_1 = pd.DataFrame(x.value, index=assets, columns=[i])
    weights = pd.concat([weights, weights_1], axis=1)

display(weights)


# CVXPY's __SOC constraint__ also solves the error that we see when we use __quad_form__ in constraints.

# In[9]:


# Calculating Annualized Portfolio Stats
var = weights * (Y.cov() @ weights) * 252
var = var.sum().to_frame().T
std = np.sqrt(var)
ret = Y.mean().to_frame().T @ weights * 252

stats = pd.concat([ret, std, var], axis=0)
stats.index = ["Return", "Std. Dev.", "Variance"]

display(stats)


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
#
# - https://www.cvxpy.org/examples/index.html
#
# </font>
# </div>

# In[ ]:
