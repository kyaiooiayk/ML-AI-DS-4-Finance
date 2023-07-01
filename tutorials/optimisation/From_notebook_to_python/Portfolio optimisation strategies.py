#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Mean-Variance-Optimization" data-toc-modified-id="Mean-Variance-Optimization-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Mean-Variance Optimization</a></span></li><li><span><a href="#Markovitz-curse" data-toc-modified-id="Markovitz-curse-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Markovitz curse</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Get-data" data-toc-modified-id="Get-data-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Get data</a></span></li><li><span><a href="#Compute-Inputs" data-toc-modified-id="Compute-Inputs-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Compute Inputs</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Compute-Returns" data-toc-modified-id="Compute-Returns-6.0.1"><span class="toc-item-num">6.0.1&nbsp;&nbsp;</span>Compute Returns</a></span></li><li><span><a href="#Set--Parameters" data-toc-modified-id="Set--Parameters-6.0.2"><span class="toc-item-num">6.0.2&nbsp;&nbsp;</span>Set  Parameters</a></span></li><li><span><a href="#Annualization-Factor" data-toc-modified-id="Annualization-Factor-6.0.3"><span class="toc-item-num">6.0.3&nbsp;&nbsp;</span>Annualization Factor</a></span></li><li><span><a href="#Compute-Mean-Returns,-Covariance-and-Precision-Matrix" data-toc-modified-id="Compute-Mean-Returns,-Covariance-and-Precision-Matrix-6.0.4"><span class="toc-item-num">6.0.4&nbsp;&nbsp;</span>Compute Mean Returns, Covariance and Precision Matrix</a></span></li><li><span><a href="#Risk-Free-Rate" data-toc-modified-id="Risk-Free-Rate-6.0.5"><span class="toc-item-num">6.0.5&nbsp;&nbsp;</span>Risk-Free Rate</a></span></li></ul></li></ul></li><li><span><a href="#Simulate-Random-Portfolios" data-toc-modified-id="Simulate-Random-Portfolios-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Simulate Random Portfolios</a></span><ul class="toc-item"><li><span><a href="#Compute-Annualize-PF-Performance" data-toc-modified-id="Compute-Annualize-PF-Performance-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>Compute Annualize PF Performance</a></span></li><li><span><a href="#Max-Sharpe-PF" data-toc-modified-id="Max-Sharpe-PF-7.2"><span class="toc-item-num">7.2&nbsp;&nbsp;</span>Max Sharpe PF</a></span></li><li><span><a href="#Compute-Efficient-Frontier" data-toc-modified-id="Compute-Efficient-Frontier-7.3"><span class="toc-item-num">7.3&nbsp;&nbsp;</span>Compute Efficient Frontier</a></span></li><li><span><a href="#Min-Volatility-Portfolio" data-toc-modified-id="Min-Volatility-Portfolio-7.4"><span class="toc-item-num">7.4&nbsp;&nbsp;</span>Min Volatility Portfolio</a></span></li><li><span><a href="#Run-Calculation" data-toc-modified-id="Run-Calculation-7.5"><span class="toc-item-num">7.5&nbsp;&nbsp;</span>Run Calculation</a></span><ul class="toc-item"><li><span><a href="#Get-random-PF" data-toc-modified-id="Get-random-PF-7.5.1"><span class="toc-item-num">7.5.1&nbsp;&nbsp;</span>Get random PF</a></span></li><li><span><a href="#Get-Max-Sharpe-PF" data-toc-modified-id="Get-Max-Sharpe-PF-7.5.2"><span class="toc-item-num">7.5.2&nbsp;&nbsp;</span>Get Max Sharpe PF</a></span></li><li><span><a href="#Get-Min-Vol-PF" data-toc-modified-id="Get-Min-Vol-PF-7.5.3"><span class="toc-item-num">7.5.3&nbsp;&nbsp;</span>Get Min Vol PF</a></span></li><li><span><a href="#Get-Efficent-PFs" data-toc-modified-id="Get-Efficent-PFs-7.5.4"><span class="toc-item-num">7.5.4&nbsp;&nbsp;</span>Get Efficent PFs</a></span></li><li><span><a href="#Plot-Result" data-toc-modified-id="Plot-Result-7.5.5"><span class="toc-item-num">7.5.5&nbsp;&nbsp;</span>Plot Result</a></span></li></ul></li></ul></li><li><span><a href="#Clean-up" data-toc-modified-id="Clean-up-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Clean-up</a></span></li><li><span><a href="#References" data-toc-modified-id="References-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Mean-Variance optimisation
# 
# </font>
# </div>

# # Mean-Variance Optimization
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - MPT solves for the optimal portfolio weights to minimize volatility for a given expected return, or maximize returns for a given level of volatility. The key requisite input are expected asset returns, standard deviations, and the covariance matrix. 
# 
# - Diversification works because the variance of portfolio returns depends on the covariance of the assets and can be reduced below the weighted average of the asset variances by including assets with less than perfect correlation. In particular, given a vector, ω, of portfolio weights and the covariance matrix, $\Sigma$, the portfolio variance, $\sigma_{\text{PF}}$ is defined as:
# $$\sigma_{\text{PF}}=\omega^T\Sigma\omega$$
# 
# - Markowitz showed that the problem of maximizing the expected portfolio return subject to a target risk has an equivalent dual representation of minimizing portfolio risk subject to a target expected return level, $μ_{PF}$. Hence, the optimization problem becomes:
# $$
# \begin{align}
# \min_\omega & \quad\quad\sigma^2_{\text{PF}}= \omega^T\Sigma\omega\\
# \text{s.t.} &\quad\quad \mu_{\text{PF}}= \omega^T\mu\\ 
# &\quad\quad \lVert\omega\rVert =1
# \end{align}
# $$
# 
# 
# </font>
# </div>

# # Markovitz curse
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - The covariance matrix can be estimated somewhat more reliably, which has given rise to several alternative approaches. However, covariance matrices with correlated assets pose computational challenges since the optimization problem requires inverting the matrix. 
# 
# - The high condition number induces numerical instability, which in turn gives rise to the **Markovitz curse**: the more diversification is required (by correlated investment opportunities), the more unreliable the weights produced by the algorithm.
# 
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[1]:


import os
import pandas as pd
import numpy as np
from numpy.random import random, uniform, dirichlet, choice
from numpy.linalg import inv

from scipy.optimize import minimize

import pandas_datareader.data as web
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from pathlib import Path


# In[2]:


sns.set_style('whitegrid')
np.random.seed(42)


# In[3]:


cmap = sns.diverging_palette(10, 240, n=9, as_cmap=True)


# # Get data
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - These are the instructions for downloading the Quandl Wiki stock prices. 
# - Step-by-step procedure:
#     1. Follow the instructions to create a free [NASDAQ account](https://data.nasdaq.com/sign-up)
#     2. If the link above does not work use [this]((https://data.nasdaq.com/databases/WIKIP/documentation)) one to sign-up 
#     3. [Download](https://data.nasdaq.com/tables/WIKIP/WIKI-PRICES/export) the entire WIKI/PRICES data
#     4. Extract the .zip file,
#     5. Move to this directory and rename to `wiki_prices.csv`
#     6. Run the below code to store in fast HDF format. 
# 
# </font>
# </div>

# In[4]:


# Check if .csv is present
get_ipython().system('ls *.csv')


# In[5]:


# read csv with pandas
df = (pd.read_csv('wiki_prices.csv',
                 parse_dates=['date'],
                 index_col=['date', 'ticker'],
                 infer_datetime_format=True)
     .sort_index())


# In[6]:


# Some sanity check on the dataframe
df.info(null_counts=True)


# In[7]:


# Put data into an h5 format
DATA_STORE = Path('assets.h5')
with pd.HDFStore(DATA_STORE) as store:
    store.put('quandl/wiki/prices', df)


# - The following code downloads the current S&P 500 constituents from [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).

# In[8]:


url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
df = pd.read_html(url, header=0)[0]


# In[9]:


df.head(5)


# In[10]:


df.columns = ['ticker', 'name', 'sec_filings', 'gics_sector', 'gics_sub_industry',
              'location', 'first_added', 'cik', 'founded']
df = df.drop('sec_filings', axis=1).set_index('ticker')


# In[11]:


print(df.info())


# In[12]:


# Saving data locally
with pd.HDFStore(DATA_STORE) as store:
    store.put('sp500/stocks', df)


# In[13]:


# Reading the data from file 
with pd.HDFStore('assets.h5') as store:
    sp500_stocks = store['sp500/stocks']


# In[14]:


sp500_stocks.head()


# In[15]:


with pd.HDFStore('assets.h5') as store:
    prices = (store['quandl/wiki/prices']
              .adj_close
              .unstack('ticker')
              .filter(sp500_stocks.index)
              .sample(n=30, axis=1))


# In[16]:


prices.head(5)


# # Compute Inputs
# <hr style = "border:2px solid black" ></hr>

# ### Compute Returns

# In[17]:


start = 2008
end = 2017


# Create month-end monthly returns and drop dates that have no observations:

# In[18]:


weekly_returns = prices.loc[f'{start}':f'{end}'].resample('W').last().pct_change().dropna(how='all')
weekly_returns = weekly_returns.dropna(axis=1)
weekly_returns.info()


# ### Set  Parameters

# In[19]:


stocks = weekly_returns.columns


# In[20]:


n_obs, n_assets = weekly_returns.shape
n_assets, n_obs


# In[21]:


NUM_PF = 100000 # no of portfolios to simulate


# In[22]:


x0 = uniform(0, 1, n_assets)
x0 /= np.sum(np.abs(x0))


# ### Annualization Factor

# In[23]:


periods_per_year = round(weekly_returns.resample('A').size().mean())
periods_per_year


# ### Compute Mean Returns, Covariance and Precision Matrix

# In[24]:


mean_returns = weekly_returns.mean()
cov_matrix = weekly_returns.cov()


# The precision matrix is the inverse of the covariance matrix:

# In[25]:


precision_matrix = pd.DataFrame(inv(cov_matrix), index=stocks, columns=stocks)


# ### Risk-Free Rate

# Load historical 10-year Treasury rate:

# In[26]:


treasury_10yr_monthly = (web.DataReader('DGS10', 'fred', start, end)
                         .resample('M')
                         .last()
                         .div(periods_per_year)
                         .div(100)
                         .squeeze())


# In[27]:


rf_rate = treasury_10yr_monthly.mean()


# # Simulate Random Portfolios
# <hr style = "border:2px solid black" ></hr>

# The simulation generates random weights using the Dirichlet distribution, and computes the mean, standard deviation, and SR for each sample portfolio using the historical return data:

# In[28]:


def simulate_portfolios(mean_ret, cov, rf_rate=rf_rate, short=True):
    alpha = np.full(shape=n_assets, fill_value=.05)
    weights = dirichlet(alpha=alpha, size=NUM_PF)
    if short:
        weights *= choice([-1, 1], size=weights.shape)

    returns = weights @ mean_ret.values + 1
    returns = returns ** periods_per_year - 1
    std = (weights @ weekly_returns.T).std(1)
    std *= np.sqrt(periods_per_year)
    sharpe = (returns - rf_rate) / std
    return pd.DataFrame({'Annualized Standard Deviation': std,
                         'Annualized Returns': returns,
                         'Sharpe Ratio': sharpe}), weights


# In[29]:


simul_perf, simul_wt = simulate_portfolios(mean_returns, cov_matrix, short=False)


# In[30]:


df = pd.DataFrame(simul_wt)
df.describe()


# In[31]:


# Plot simulated portfolios
ax = simul_perf.plot.scatter(x=0, y=1, c=2, cmap='Blues',
                             alpha=0.5, figsize=(14, 9), colorbar=True,
                             title=f'{NUM_PF:,d} Simulated Portfolios')

max_sharpe_idx = simul_perf.iloc[:, 2].idxmax()
sd, r = simul_perf.iloc[max_sharpe_idx, :2].values
print(f'Max Sharpe: {sd:.2%}, {r:.2%}')
ax.scatter(sd, r, marker='*', color='darkblue', s=500, label='Max. Sharpe Ratio')

min_vol_idx = simul_perf.iloc[:, 0].idxmin()
sd, r = simul_perf.iloc[min_vol_idx, :2].values
ax.scatter(sd, r, marker='*', color='green', s=500, label='Min Volatility')
plt.legend(labelspacing=1, loc='upper left')
plt.tight_layout()


# ## Compute Annualize PF Performance

# Now we'll set up the quadratic optimization problem to solve for the minimum standard deviation for a given return or the maximum SR. 
# 
# To this end, define the functions that measure the key metrics:

# In[32]:


def portfolio_std(wt, rt=None, cov=None):
    """Annualized PF standard deviation"""
    return np.sqrt(wt @ cov @ wt * periods_per_year)


# In[33]:


def portfolio_returns(wt, rt=None, cov=None):
    """Annualized PF returns"""
    return (wt @ rt + 1) ** periods_per_year - 1


# In[34]:


def portfolio_performance(wt, rt, cov):
    """Annualized PF returns & standard deviation"""
    r = portfolio_returns(wt, rt=rt)
    sd = portfolio_std(wt, cov=cov)
    return r, sd


# ## Max Sharpe PF

# Define a target function that represents the negative SR for scipy's minimize function to optimize, given the constraints that the weights are bounded by [-1, 1], if short trading is permitted, and [0, 1] otherwise, and sum to one in absolute terms.

# In[35]:


def neg_sharpe_ratio(weights, mean_ret, cov):
    r, sd = portfolio_performance(weights, mean_ret, cov)
    return -(r - rf_rate) / sd


# In[36]:


weight_constraint = {'type': 'eq', 
                     'fun': lambda x: np.sum(np.abs(x))-1}


# In[37]:


def max_sharpe_ratio(mean_ret, cov, short=False):
    return minimize(fun=neg_sharpe_ratio,
                    x0=x0,
                    args=(mean_ret, cov),
                    method='SLSQP',
                    bounds=((-1 if short else 0, 1),) * n_assets,
                    constraints=weight_constraint,
                    options={'tol':1e-10, 'maxiter':1e4})


# ## Compute Efficient Frontier

# The solution requires iterating over ranges of acceptable values to identify optimal risk-return combinations

# In[38]:


def min_vol_target(mean_ret, cov, target, short=False):

    def ret_(wt):
        return portfolio_returns(wt, mean_ret)

    constraints = [{'type': 'eq',
                    'fun': lambda x: ret_(x) - target},
                   weight_constraint]

    bounds = ((-1 if short else 0, 1),) * n_assets
    return minimize(portfolio_std,
                    x0=x0,
                    args=(mean_ret, cov),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'tol': 1e-10, 'maxiter': 1e4})


# The mean-variance frontier relies on in-sample, backward-looking optimization. In practice, portfolio optimization requires forward-looking input. Unfortunately, expected returns are notoriously difficult to estimate accurately.
# 
# The covariance matrix can be estimated somewhat more reliably, which has given rise to several alternative approaches. However, covariance matrices with correlated assets pose computational challenges since the optimization problem requires inverting the matrix. The high condition number induces numerical instability, which in turn gives rise to Markovitz curse: the more diversification is required (by correlated investment opportunities), the more unreliable the weights produced by the algorithm. 

# ## Min Volatility Portfolio

# In[39]:


def min_vol(mean_ret, cov, short=False):
    bounds = ((-1 if short else 0, 1),) * n_assets

    return minimize(fun=portfolio_std,
                    x0=x0,
                    args=(mean_ret, cov),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=weight_constraint,
                    options={'tol': 1e-10, 'maxiter': 1e4})


# In[40]:


def efficient_frontier(mean_ret, cov, ret_range, short=False):
    return [min_vol_target(mean_ret, cov, ret) for ret in ret_range]


# ## Run Calculation

# ### Get random PF

# In[41]:


simul_perf, simul_wt = simulate_portfolios(mean_returns, cov_matrix, short=False)


# In[42]:


print(simul_perf.describe())


# In[43]:


simul_max_sharpe = simul_perf.iloc[:, 2].idxmax()
simul_perf.iloc[simul_max_sharpe]


# ### Get Max Sharpe PF

# In[44]:


max_sharpe_pf = max_sharpe_ratio(mean_returns, cov_matrix, short=False)
max_sharpe_perf = portfolio_performance(max_sharpe_pf.x, mean_returns, cov_matrix)


# In[45]:


r, sd = max_sharpe_perf
pd.Series({'ret': r, 'sd': sd, 'sr': (r-rf_rate)/sd})


# From simulated pf data

# ### Get Min Vol PF

# In[46]:


min_vol_pf = min_vol(mean_returns, cov_matrix, short=False)
min_vol_perf = portfolio_performance(min_vol_pf.x, mean_returns, cov_matrix)


# ### Get Efficent PFs

# In[47]:


ret_range = np.linspace(simul_perf.iloc[:, 1].min(), simul_perf.iloc[:, 1].max(), 50)
eff_pf = efficient_frontier(mean_returns, cov_matrix, ret_range, short=True)
eff_pf = pd.Series(dict(zip([p['fun'] for p in eff_pf], ret_range)))


# ### Plot Result

# The simulation yields a subset of the feasible portfolios, and the efficient frontier identifies the optimal in-sample return-risk combinations that were achievable given historic data. 
# 
# The below figure shows the result including the minimum variance portfolio and the portfolio that maximizes the SR and several portfolios produce by alternative optimization strategies. The efficient frontier 

# In[48]:


fig, ax = plt.subplots()
simul_perf.plot.scatter(x=0, y=1, c=2, ax=ax, cmap='Blues',alpha=0.25, 
                        figsize=(14, 9), colorbar=True)

eff_pf[eff_pf.index.min():].plot(linestyle='--', lw=2, ax=ax, c='k',
                                 label='Efficient Frontier')

r, sd = max_sharpe_perf
ax.scatter(sd, r, marker='*', color='k', s=500, label='Max Sharpe Ratio PF')

r, sd = min_vol_perf
ax.scatter(sd, r, marker='v', color='k', s=200, label='Min Volatility PF')

kelly_wt = precision_matrix.dot(mean_returns).clip(lower=0).values
kelly_wt /= np.sum(np.abs(kelly_wt))
r, sd = portfolio_performance(kelly_wt, mean_returns, cov_matrix)
ax.scatter(sd, r, marker='D', color='k', s=150, label='Kelly PF')

std = weekly_returns.std()
std /= std.sum()
r, sd = portfolio_performance(std, mean_returns, cov_matrix)
ax.scatter(sd, r, marker='X', color='k', s=250, label='Risk Parity PF')

r, sd = portfolio_performance(np.full(n_assets, 1/n_assets), mean_returns, cov_matrix)
ax.scatter(sd, r, marker='o', color='k', s=200, label='1/n PF')


ax.legend(labelspacing=0.8)
ax.set_xlim(0, eff_pf.max()+.4)
ax.set_title('Mean-Variance Efficient Frontier', fontsize=16)
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
sns.despine()
fig.tight_layout();


# # Clean-up
# <hr style = "border:2px solid black" ></hr>

# In[49]:


#1.54Gb
os.remove("./assets.h5")


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - [How to download the data](https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/data/create_datasets.ipynb)
# - [This notebook original code](https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/05_strategy_evaluation/04_mean_variance_optimization.ipynb)
# - Jansen, Stefan. Hands-On Machine Learning for Algorithmic Trading: Design and implement investment strategies based on smart algorithms that learn from data using Python. Packt Publishing Ltd, 2018.
# 
# </font>
# </div>

# In[ ]:




