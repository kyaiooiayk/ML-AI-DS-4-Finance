#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#ARCH/GARCH-Volatility-Forecasting" data-toc-modified-id="ARCH/GARCH-Volatility-Forecasting-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>ARCH/GARCH Volatility Forecasting</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Download-data" data-toc-modified-id="Download-data-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Download data</a></span></li><li><span><a href="#Correlogram" data-toc-modified-id="Correlogram-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Correlogram</a></span></li><li><span><a href="#Model-Selection:-rolling-out-of-sample-forecasts" data-toc-modified-id="Model-Selection:-rolling-out-of-sample-forecasts-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Model Selection: rolling out-of-sample forecasts</a></span></li><li><span><a href="#Estimate-GARCH(2,-2)-Model" data-toc-modified-id="Estimate-GARCH(2,-2)-Model-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Estimate GARCH(2, 2) Model</a></span></li><li><span><a href="#Check-Residuals" data-toc-modified-id="Check-Residuals-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Check Residuals</a></span></li><li><span><a href="#References" data-toc-modified-id="References-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** ARCH/GARCH Volatility Forecasting
# 
# </font>
# </div>

# # ARCH/GARCH Volatility Forecasting

# The development of a volatility model for an asset-return series consists of four steps:
# 1. Build an ARMA time series model for the financial time series based on the serial dependence revealed by the ACF and PACF.
# 2. Test the residuals of the model for ARCH/GARCH effects, again relying on the ACF and PACF for the series of the squared residual.
# 3. Specify a volatility model if serial correlation effects are significant, and jointly estimate the mean and volatility equations.
# 4. Check the fitted model carefully and refine it if necessary
# 
# When applying volatility forecasting to return series, the serial dependence may be limited so that a constant mean may be used instead of an ARMA model.

# The `arch` library provides several options to estimate volatility-forecasting models. It offers several options to model the expected mean, including a constant mean, the AR(p) model discussed in the section on univariate time series models above as well as more recent heterogeneous autoregressive processes (HAR) that use daily (1 day), weekly (5 days), and monthly (22 days) lags to capture the trading frequencies of short-, medium-, and long-term investors.
# 
# The mean models can be jointly defined and estimated with several conditional heteroskedasticity models that include, in addition to ARCH and GARCH, the exponential GARCH (EGARCH) model, which allows for asymmetric effects between positive and negative returns and the heterogeneous ARCH (HARCH) model, which complements the HAR mean model.

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd
import pandas_datareader.data as web

import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, q_stat, adfuller

from sklearn.metrics import mean_squared_error

from scipy.stats import probplot, moment

from arch import arch_model
from arch.univariate import ConstantMean, GARCH, Normal

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm


# In[3]:


sns.set_style('whitegrid')


# # Download data
# <hr style = "border:2px solid black" ></hr>

# We will use daily NASDAQ returns from 1998-2017 to demonstrate the usage of a GARCH model

# In[5]:


nasdaq = web.DataReader('NASDAQCOM', 'fred', '2000', '2020').squeeze()
nasdaq_returns = np.log(nasdaq).diff().dropna().mul(
    100)  # rescale to faciliate optimization


# # Correlogram
# <hr style = "border:2px solid black" ></hr>

# In[23]:


def plot_correlogram(x, lags=None, title=None):
    lags = min(10, int(len(x)/5)) if lags is None else lags
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
    x.plot(ax=axes[0][0], title='Residuals')
    x.rolling(21).mean().plot(ax=axes[0][0], c='k', lw=1)
    q_p = np.max(q_stat(acf(x, nlags=lags), len(x))[1])
    
    stats = f'Q-Stat: {np.max(q_p):>8.2f}\nADF: {adfuller(x)[1]:>11.2f}'
    axes[0][0].text(x=.02, y=.85, s=stats, transform=axes[0][0].transAxes)
    probplot(x, plot=axes[0][1])
    mean, var, skew, kurtosis = moment(x, moment=[1, 2, 3, 4])
    s = f'Mean: {mean:>12.2f}\nSD: {np.sqrt(var):>16.2f}\nSkew: {skew:12.2f}\nKurtosis:{kurtosis:9.2f}'
    
    axes[0][1].text(x=.02, y=.75, s=s, transform=axes[0][1].transAxes)
    
    plot_acf(x=x, lags=lags, zero=False, ax=axes[1][0])
    plot_pacf(x, lags=lags, zero=False, ax=axes[1][1])
    axes[1][0].set_xlabel('Lag')
    axes[1][1].set_xlabel('Lag')
    
    axes[1][0].set_ylim([-0.06,0.06])
    axes[1][1].set_ylim([-0.06,0.06])
        
    fig.suptitle(title, fontsize=14)
    sns.despine()
    fig.tight_layout()
    fig.subplots_adjust(top=.9)


# The rescaled daily return series exhibits only limited autocorrelation, but the squared deviations from the mean do have substantial memory reflected in the slowly-decaying ACF and the PACF high for the first two and cutting off only after the first six lags:

# In[24]:


plot_correlogram(nasdaq_returns, lags=250, title='NASDAQ Daily Returns')


# The function plot_correlogram produces the following output:

# In[25]:


plot_correlogram(nasdaq_returns.sub(nasdaq_returns.mean()).pow(
    2), lags=120, title='NASDAQ Daily Volatility')


# # Model Selection: rolling out-of-sample forecasts
# <hr style = "border:2px solid black" ></hr>

# Hence, we can estimate a GARCH model to capture the linear relationship of past volatilities. We will use rolling 10-year windows to estimate a GARCH(p, q) model with p and q ranging from 1-4 to generate 1-step out-of-sample forecasts. 
# 
# We then compare the RMSE of the predicted volatility relative to the actual squared deviation of the return from its mean to identify the most predictive model. We are using winsorized data to limit the impact of extreme return values reflected in the very high positive skew of the volatility

# In[8]:


trainsize = 10 * 252  # 10 years
data = nasdaq_returns.clip(lower=nasdaq_returns.quantile(.05),
                           upper=nasdaq_returns.quantile(.95))
T = len(nasdaq_returns)
results = {}
"""3 just to save time"""
for p in range(1, 3):
    for q in range(1, 3):
        print(f'{p} | {q}')
        result = []
        for s, t in tqdm(enumerate(range(trainsize, T-1))):
            train_set = data.iloc[s: t]
            test_set = data.iloc[t+1]  # 1-step ahead forecast
            model = arch_model(y=train_set, p=p, q=q).fit(disp='off')
            forecast = model.forecast(horizon=1)
            mu = forecast.mean.iloc[-1, 0]
            var = forecast.variance.iloc[-1, 0]
            result.append([(test_set-mu)**2, var])
        df = pd.DataFrame(result, columns=['y_true', 'y_pred'])
        results[(p, q)] = np.sqrt(mean_squared_error(df.y_true, df.y_pred))


# In[9]:


s = pd.Series(results)
s.index.names = ['p', 'q']
s = s.unstack().sort_index(ascending=False)


# In[10]:


sns.heatmap(s, cmap='Blues', annot=True, fmt='.4f')
plt.title('Out-of-Sample RMSE');


# # Estimate GARCH(2, 2) Model
# <hr style = "border:2px solid black" ></hr>

# The GARCH(2, 2) model achieves the second-lowest RMSE but has fewer parameters than the slightly better-performing GARCH(2, 4), so we go ahead and estimate this model to inspect the summary.

# The output shows the maximized log-likelihood as well as the AIC and BIC criteria that are commonly minimized when selecting models based on in-sample performance (see Chapter 7, Linear Models). It also displays the result for the mean model, which in this case is just a constant estimate, as well as the GARCH parameters for the constant omega, the AR parameters, α, and the MA parameters, β, all of which are statistically significant:

# In[11]:


best_p, best_q = 2, 2,


# In[12]:


am = ConstantMean(nasdaq_returns.clip(lower=nasdaq_returns.quantile(.05),
                                      upper=nasdaq_returns.quantile(.95)))
am.volatility = GARCH(best_p, 0, best_q)
am.distribution = Normal()
best_model = am.fit(update_freq=5)
print(best_model.summary())


# # Check Residuals
# <hr style = "border:2px solid black" ></hr>

# In[13]:


fig = best_model.plot(annualize='D')
fig.set_size_inches(12, 8)
fig.tight_layout();


# In[14]:


plot_correlogram(best_model.resid.dropna(),
                 lags=250,
                 title='GARCH Residuals')


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - [This notebook code](https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/09_time_series_models/03_arch_garch_models.ipynb)
# - Wilmott, Paul. Frequently asked questions in quantitative finance. John Wiley & Sons, 2010. 
# 
# </font>
# </div>

# In[ ]:




