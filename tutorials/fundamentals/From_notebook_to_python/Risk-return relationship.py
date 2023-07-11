#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Risk-return" data-toc-modified-id="Risk-return-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Risk-return</a></span></li><li><span><a href="#Generate-data" data-toc-modified-id="Generate-data-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Generate data</a></span></li><li><span><a href="#Fit-the-data" data-toc-modified-id="Fit-the-data-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Fit the data</a></span></li><li><span><a href="#Conclusion" data-toc-modified-id="Conclusion-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Conclusion</a></span></li><li><span><a href="#References" data-toc-modified-id="References-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
#
# **What?** Risk-return relationship
#
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[2]:


import statsmodels.api as sm
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly
import warnings

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300


# # Risk-return
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# - There is a trade-off between risk and return: the higher the assumed risk, the greater the realized return.
#
# - As it is a formidable task to come up with an optimum solution, this trade-off is one of the most controversial issues in finance.
#
# - However, **Markowitz** (1952) proposes an intuitive and appealing solution to this long-standing issue. He used standard deviation to quantify risk.
#
#
# </font>
# </div>

# # Generate data
# <hr style = "border:2px solid black" ></hr>

# - An hypothetical portfolio is constructed to calculate necessary statistics: standard deviation and covariance

# In[3]:


n_assets = 5
n_simulation = 500


# In[9]:


returns = np.random.randn(n_assets, n_simulation)


# In[10]:


rand = np.random.rand(n_assets)
weights = rand / sum(rand)


def port_return(returns):
    """
    Function used to calculate expected portfolio
    return and portfolio standard deviation
    """
    rets = np.mean(returns, axis=1)
    cov = np.cov(rets.T, aweights=weights, ddof=1)
    portfolio_returns = np.dot(weights, rets.T)
    portfolio_std_dev = np.sqrt(np.dot(weights, np.dot(cov, weights)))
    return portfolio_returns, portfolio_std_dev


# In[11]:


portfolio_returns, portfolio_std_dev = port_return(returns)


# In[12]:


print(portfolio_returns)
print(portfolio_std_dev)


# In[13]:


portfolio = np.array([port_return(np.random.randn(n_assets, i)) for i in range(1, 101)])


# # Fit the data
# <hr style = "border:2px solid black" ></hr>

# In[14]:


best_fit = sm.OLS(portfolio[:, 1], sm.add_constant(portfolio[:, 0])).fit().fittedvalues


# In[15]:


fig = go.Figure()
fig.add_trace(
    go.Scatter(
        name="Risk-Return Relationship",
        x=portfolio[:, 0],
        y=portfolio[:, 1],
        mode="markers",
    )
)
fig.add_trace(
    go.Scatter(name="Best Fit Line", x=portfolio[:, 0], y=best_fit, mode="lines")
)
fig.update_layout(
    xaxis_title="Return", yaxis_title="Standard Deviation", width=900, height=470
)
fig.show()


# # Conclusion
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-danger">
# <font color=black>
#
# - The plot confirms that the risk and return go in tandem.
#
# - However, keep in mind that the magnitude of this correlation varies depending on the individual stock and the financial market conditions.
#
#
# </font>
# </div>

# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
#
# - https://github.com/abdullahkarasan/mlfrm/blob/main/codes/chp_1.ipynb
# - Machine Learning for Financial Risk Management with Python Abdullah Karasan
#
# </font>
# </div>
