#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Objective" data-toc-modified-id="Objective-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Objective</a></span></li><li><span><a href="#Volatility" data-toc-modified-id="Volatility-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Volatility</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Get-the-data" data-toc-modified-id="Get-the-data-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Get the data</a></span></li><li><span><a href="#References" data-toc-modified-id="References-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Bayesian Stochastic Volatility Model
# 
# </font>
# </div>

# # Objective
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#  
# - A Bayesian stochastic volatility model provides a full posterior probability distribution of volatility at each time point t, as opposed to a single "point estimate" often provided by other models. 
# 
# - This posterior encapsulates the uncertainty in the parameters and can be used to obtain credible intervals (analogous to confidence intervals) and other statistics about the volatility.
# 
# </font>
# </div>

# # Volatility
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - **Markowitz** proposed his celebrated portfolio theory in which he defined volatility as standard deviation so that from then onward, finance became more intertwined with mathematics.
# 
# - Modeling volatility amounts to modeling uncertainty so that we better understand and approach uncertainty, enabling us to have good enough approximations of the real world. 
#     
# - To gauge the extent to which proposed models account for the real-world situation, we need to calculate the **return volatility**, which is also known as **realised volatility**.
# 
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[5]:


import datetime 
import pprint
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import pandas_datareader as pdr 
import pymc3 as pm
from pymc3.distributions.timeseries import GaussianRandomWalk 
import seaborn as sns
import pandas as pd
import pandas_datareader as pdr


# # Get the data
# <hr style = "border:2px solid black" ></hr>

# In[10]:


# year-month-day format

# State the starting and ending dates of the AMZN returns 
start_date = datetime.datetime(2006, 1, 1)
end_date = datetime.datetime(2015, 12, 31)
# Obtain and plot the logarithmic returns of Amazon prices
amzn_df = pdr.get_data_yahoo("AMZN", start_date, end_date)

amzn_df["returns"] = amzn_df["Adj Close"]/amzn_df["Adj Close"].shift(1) 
amzn_df.dropna(inplace=True)
amzn_df["log_returns"] = np.log(amzn_df["returns"]) 

log_returns = np.array(amzn_df["log_returns"])


# In[11]:


amzn_df["log_returns"].plot(linewidth=0.5, figsize =(8,5))


# <div class="alert alert-info">
# <font color=black>
#  
# - Bayesian formulation of a stochastic volatility model, the sampling of which is carried out using the presented NUTS technique.
# - No-U-Turn Sampler (NUTS) is a highly efficient form of Markov Chain Monte Carlo.
# 
# - To define the prior we need to define 
#     - sigma $σ$, which represents the scale of the volatility
#     - nu $ν$ which represents the degrees of freedom of the Student’s t-distribution. 
#     
# - Priors must also be selected for the latent volatility process and subsequently the asset returns distribution.
# At this stage the uncertainty on the parameter values is large and so the selected priors must reflect that. In addition σ and ν must be real-valued positive numbers, so we need to use a probability distribution that has positive support. As an initial choice the exponential distribution will be chosen for σ and ν.
# 
# - A much larger parameter value is chosen for σ than ν because there is a lot of initial uncertainty associated with the scale of the volatility generating process. As more data is provided to the model the Bayesian updating process will reduce the spread of the posterior distribution reflecting an increased certainty in the scale factor of volatility.
#     
# - This stochastic volatility model makes use of a random walk model for the latent volatility variable. Random walk models are discussed in significant depth within the subsequent time series chapter on White Noise and Random Walks.
#     
# - It remains only to assign a prior to the logarithmic returns of the asset price series being modelled. The point of a stochastic volatility model is that these returns are related to the underlying latent volatility variable. Hence any prior that is assigned to the log returns must have a variance that involves s. One approach (is to assume that the log returns are distributed as a Student’s t-distribution.
#     
# </font>
# </div>

# In[22]:


samples = 2000 
model = pm.Model()
with model:

    sigma = pm.Exponential("sigma", 50.0, testval=0.1)

    nu = pm.Exponential("nu", 0.1)

    s = GaussianRandomWalk("s", sigma**-2, shape=len(log_returns))

    logrets = pm.StudentT("logrets", nu,
                          lam=pm.math.exp(
                              -2.0*s),
                          observed=log_returns
                          )


# In[ ]:





# In[23]:


# print("Fitting the stochastic volatility model...")
with model:
    trace = pm.sample(samples)


# In[ ]:


pm.traceplot(trace, model.vars[:-1])
plt.show()


# In[ ]:


print("Plotting the log volatility...")
k = 10
opacity = 0.03
plt.plot(trace[s][::k].T, "b", alpha=opacity)
plt.xlabel("Time")
plt.ylabel("Log Volatility")
plt.show()


# In[19]:


print("Plotting the absolute returns overlaid with vol...")
plt.plot(np.abs(np.exp(log_returns))-1.0, linewidth=0.5)
plt.plot(np.exp(trace[s][::k].T), "r", alpha=opacity)
plt.xlabel("Trading Days")
plt.ylabel("Absolute Returns/Volatility")
plt.show()


# In[20]:


# Configure the stochastic volatility model and carry out # MCMC sampling using NUTS, plotting the trace

configure_sample_stoch_vol_model(log_returns, samples)


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://www.quantstart.com/advanced-algorithmic-trading-ebook/ 
# 
# </font>
# </div>

# In[ ]:




