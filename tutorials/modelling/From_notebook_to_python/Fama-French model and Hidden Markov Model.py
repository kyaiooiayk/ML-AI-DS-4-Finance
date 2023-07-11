#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1">Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2">Imports</a></span></li><li><span><a href="#HMM---Hidden-Markov-Model" data-toc-modified-id="HMM---Hidden-Markov-Model-3">HMM - Hidden Markov Model</a></span></li><li><span><a href="#Fame-French-model" data-toc-modified-id="Fame-French-model-4">Fame-French model</a></span></li><li><span><a href="#Get-the-data" data-toc-modified-id="Get-the-data-5">Get the data</a></span></li><li><span><a href="#Selecting-a-stock-return" data-toc-modified-id="Selecting-a-stock-return-6">Selecting a stock-return</a></span></li><li><span><a href="#Gaussian-HMM" data-toc-modified-id="Gaussian-HMM-7">Gaussian HMM</a></span></li><li><span><a href="#Optimal-number-of-hidden-states---elbow-analysis" data-toc-modified-id="Optimal-number-of-hidden-states---elbow-analysis-8">Optimal number of hidden states - elbow analysis</a></span></li><li><span><a href="#Visualisation-of-the-states" data-toc-modified-id="Visualisation-of-the-states-9">Visualisation of the states</a></span></li><li><span><a href="#Comparison" data-toc-modified-id="Comparison-10">Comparison</a></span><ul class="toc-item"><li><span><a href="#Fama-French-Model-vs.-HMM" data-toc-modified-id="Fama-French-Model-vs.-HMM-10.1">Fama-French Model vs. HMM</a></span></li><li><span><a href="#Fama-French-Model-with-OLS" data-toc-modified-id="Fama-French-Model-with-OLS-10.2">Fama-French Model with OLS</a></span></li><li><span><a href="#Backtesting" data-toc-modified-id="Backtesting-10.3">Backtesting</a></span></li><li><span><a href="#Synthetic-Data-Generation-and-Hidden-Markov" data-toc-modified-id="Synthetic-Data-Generation-and-Hidden-Markov-10.4">Synthetic Data Generation and Hidden Markov</a></span></li></ul></li><li><span><a href="#References" data-toc-modified-id="References-11">References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
#
# **What?** Synthetic data generation
#
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
from matplotlib import rcParams
from hmmlearn import hmm
from matplotlib.dates import YearLocator, MonthLocator
from matplotlib import cm
import statsmodels.api as sm

# %matplotlib notebook
rcParams["figure.figsize"] = 6, 6
rcParams["font.size"] = 20


# In[2]:


import warnings

warnings.filterwarnings("ignore")


# # HMM - Hidden Markov Model
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
#
# - HMM gives us a probability distribution over sequential data, which is modeled by a Markov process with hidden states. HMM enables us to estimate probability transition from one state to another.
#
# - To illustrate, let us consider the stock market, in which stocks go up, go down, or stay constant. Pick a random state—say, going up. The next state would be either going up, going down, or staying constant. In this context, the state is thought to be a hidden state because we do not know with certainty which state will prevail next in the market.
#
# - HMM makes two assumptions:
#     - All observations are solely dependent on the current state and are conditionally independent of other variables
#     - The transition probabilities are homogenous and depend only on the current hidden state
#
# </font>
# </div>

# # Fame-French model
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# - The Fama-French model expands on the previous CAPM model. The model suggests  3 brand-new explanatory variables/factors to account for the changes in stock returns.
#
#     - [x] **market risk Rm − Rf** - is basically the return of a market portfolio minus the risk-free rate, which is a hypothetical rate proxied by government-issued T-bills or similar assets.
#     - [x] **small minus big (SMB)** - is a proxy for size effect. Size effect is an important variable used to explain several phenomenon in corporate finance. It is represented by different variables like logarithm of total assets. Fama-French takes size effect into account by calculating between returns of small-cap companies and big-cap companies.
#     - [x] **high minus low (HML)** - represents the spread in returns between companies with high book-to-market and companies with low book-to-market, comparing a company’s book value to its market value.
#
# - Empirical studies suggest that smaller SMB, higher HML, and smaller Rm−Rf boosts stock returns.
#
# </font>
# </div>

# # Get the data
# <hr style = "border:2px solid black" ></hr>

# In[3]:


ff = pd.read_csv("./FF3.csv", skiprows=4)
ff = ff.rename(columns={"Unnamed: 0": "Date"})
ff = ff.iloc[:-1]
ff.head()


# In[4]:


ff.info()


# In[5]:


ff["Date"] = pd.to_datetime(ff["Date"])
ff.set_index("Date", inplace=True)
ff_trim = ff.loc["2000-01-01":]


# In[6]:


ff_trim.head()


# # Selecting a stock-return
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# - ETF is a special type of investment fund and exchange-traded product that tracks industry, commodities, and so on.
# - SPDR S&P 500 ETF (**SPY**) is a very well-known example tracking the S&P 500 Index.
#
# </font>
# </div>

# In[7]:


ticker = "SPY"
start = datetime.datetime(2000, 1, 3)
end = datetime.datetime(2021, 4, 30)
SP_ETF = yf.download(ticker, start, end, interval="1d").Close


# In[8]:


ff_merge = pd.merge(ff_trim, SP_ETF, how="inner", on="Date")


# In[9]:


SP = pd.DataFrame()
SP["Close"] = ff_merge["Close"]


# In[10]:


SP["return"] = (SP["Close"] / SP["Close"].shift(1)) - 1


# In[11]:


ff_merge


# # Gaussian HMM
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# - The data including return and volatility are used to define the the hidden states.
# - It is assumed that there are three states in the economy: up, down, and constant.
# - With this in mind, we run the HMM with full covariance, indicating independent components and a number of iterations (n_iter) of 100.
# - The Gaussian HMM is used to predict the hidden state.
#
# </font>
# </div>

# In[12]:


hmm_model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)


# In[13]:


# Fitting the Gaussian HMM with return data
hmm_model.fit(np.array(SP["return"].dropna()).reshape(-1, 1))

# Given the return data, predicting the hidden states
hmm_predict = hmm_model.predict(np.array(SP["return"].dropna()).reshape(-1, 1))

df_hmm = pd.DataFrame(hmm_predict)


# In[14]:


ret_merged = pd.concat([df_hmm, SP["return"].dropna().reset_index()], axis=1)
ret_merged.drop("Date", axis=1, inplace=True)
ret_merged.rename(columns={0: "states"}, inplace=True)
ret_merged.dropna().head()


# In[15]:


ret_merged["states"].value_counts()


# In[16]:


state_means = []
state_std = []

for i in range(3):
    state_means.append(ret_merged[ret_merged.states == i]["return"].mean())
    state_std.append(ret_merged[ret_merged.states == i]["return"].std())
print("State Means are: {}".format(state_means))
print("State Standard Deviations are: {}".format(state_std))


# In[17]:


print(f"HMM means\n {hmm_model.means_}")
print(f"HMM covariances\n {hmm_model.covars_}")
print(f"HMM transition matrix\n {hmm_model.transmat_}")
print(f"HMM initial probability\n {hmm_model.startprob_}")


# # Optimal number of hidden states - elbow analysis
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# -  We assume that the economy has three states, but this assumption rests on theory.
# - However we never know if this is the case, so which is the optimal number of hidden states?
# - **Elbow Analysis** will help us understand the optimal number of hidden states. After running Gaussian HMM, we obtain the likelihood result, and if there is no room for improvement—that is, the likelihood value becomes relatively stagnant, then this is the point at which we can stop the analysis.
# - Given the following result it turns out that three components is a good choice.
#
# </font>
# </div>

# In[18]:


sp_ret = SP["return"].dropna().values.reshape(-1, 1)
n_components = np.arange(1, 10)
clusters = [
    hmm.GaussianHMM(n_components=n, covariance_type="full").fit(sp_ret)
    for n in n_components
]
plt.plot(
    n_components,
    [m.score(np.array(SP["return"].dropna()).reshape(-1, 1)) for m in clusters],
)
plt.title("Optimum Number of States")
plt.xlabel("n_components")
plt.ylabel("Log Likelihood")


# # Visualisation of the states
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
#
# - We are going to plot the the hidden state distributions.
# - We can see how these are entirely different from each other, highlighting the importance of identifying the states.
#
# </font>
# </div>

# In[19]:


hmm_model = hmm.GaussianHMM(
    n_components=3, covariance_type="full", random_state=123
).fit(sp_ret)
hidden_states = hmm_model.predict(sp_ret)


# In[20]:


df_sp_ret = SP["return"].dropna()

hmm_model = hmm.GaussianHMM(
    n_components=3, covariance_type="full", random_state=123
).fit(sp_ret)

hidden_states = hmm_model.predict(sp_ret)

fig, axs = plt.subplots(
    hmm_model.n_components, sharex=True, sharey=True, figsize=(12, 9)
)
colors = cm.gray(np.linspace(0, 0.7, hmm_model.n_components))

for i, (ax, color) in enumerate(zip(axs, colors)):
    mask = hidden_states == i
    ax.plot_date(df_sp_ret.index.values[mask], df_sp_ret.values[mask], ".-", c=color)
    ax.set_title("Hidden state {}".format(i + 1), fontsize=16)
    ax.xaxis.set_minor_locator(MonthLocator())
plt.tight_layout()


# In[21]:


ret_merged.groupby("states")["return"].mean()


# # Comparison
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# - We can move forward and run the Fama-Frech three-factor model with and without Gaussian HMM.
# - The Sharpe ratio, which we’ll calculate after modeling, will tell us which is the better risk-adjusted return.
#
# </font>
# </div>

# ## Fama-French Model vs. HMM

# In[22]:


ff_merge["return"] = ff_merge["Close"].pct_change()
ff_merge.dropna(inplace=True)


# In[23]:


split = int(len(ff_merge) * 0.9)
train_ff = ff_merge.iloc[:split].dropna()
test_ff = ff_merge.iloc[split:].dropna()


# In[24]:


hmm_model = hmm.GaussianHMM(
    n_components=3, covariance_type="full", n_iter=100, init_params=" "
)


# In[25]:


predictions = []

for i in range(len(test_ff)):
    hmm_model.fit(train_ff)
    adjustment = np.dot(hmm_model.transmat_, hmm_model.means_)
    predictions.append(test_ff.iloc[i] + adjustment[0])
predictions = pd.DataFrame(predictions)


# In[26]:


std_dev = predictions["return"].std()
sharpe = predictions["return"].mean() / std_dev
print("Sharpe ratio with HMM is {:.4f}".format(sharpe))


# ## Fama-French Model with OLS

# <div class="alert alert-info">
# <font color=black>
#
# - The traditional way to run Fama-Frech three-factor model is to apply linear regression.
#
# </font>
# </div>

# In[27]:


Y = train_ff["return"]
X = train_ff[["Mkt-RF", "SMB", "HML"]]


# In[28]:


model = sm.OLS(Y, X)
ff_ols = model.fit()
print(ff_ols.summary())


# In[29]:


ff_pred = ff_ols.predict(test_ff[["Mkt-RF", "SMB", "HML"]])
ff_pred.head()


# In[30]:


std_dev = ff_pred.std()
sharpe = ff_pred.mean() / std_dev
print("Sharpe ratio with FF 3 factor model is {:.4f}".format(sharpe))


# ## Backtesting

#
# <div class="alert alert-info">
# <font color=black>
#
# - The following analysis tries to show what happens if the states of the index return need to be predicted based on the unseen data that can be used for backtesting for future analysis.
#
# </font>
# </div>
#

# In[32]:


split = int(len(SP["return"]) * 0.9)
train_ret_SP = SP["return"].iloc[split:].dropna()
test_ret_SP = SP["return"].iloc[:split].dropna()


# In[33]:


hmm_model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
hmm_model.fit(np.array(train_ret_SP).reshape(-1, 1))
hmm_predict_vol = hmm_model.predict(np.array(test_ret_SP).reshape(-1, 1))
pd.DataFrame(hmm_predict_vol).value_counts()


# ## Synthetic Data Generation and Hidden Markov

# <div class="alert alert-info">
# <font color=black>
#
# - HMM provides a helpful and strong way for further expanding our analysis to get more reliable and accurate results.
#
# - Gaussina HHM can be used as a synthetic data generation process using Gaussian HMM. To do that, we should first define our initial parameters. These parameters are: initial probability (startprob), transition matrix (transmat), mean (means), and covariance (covars).
#
# - We can then run Gaussian HMM and apply a random sampling procedure to end up with a desired number of observations, which is 1,000 in our case.
#
# </font>
# </div>

# In[34]:


startprob = hmm_model.startprob_
transmat = hmm_model.transmat_
means = hmm_model.means_
covars = hmm_model.covars_


# In[35]:


syn_hmm = hmm.GaussianHMM(n_components=3, covariance_type="full")


# In[36]:


syn_hmm.startprob_ = startprob
syn_hmm.transmat_ = transmat
syn_hmm.means_ = means
syn_hmm.covars_ = covars


# In[37]:


syn_data, _ = syn_hmm.sample(n_samples=1000)


# In[38]:


plt.hist(syn_data)
plt.title("Histogram of Synthetic Data")
plt.show()


# In[39]:


plt.plot(syn_data, "--")
plt.title("Line Plot of Synthetic Data")
plt.show()


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
#
# - https://github.com/abdullahkarasan/mlfrm/blob/main/codes/chp_10.ipynb
# - Machine Learning for Financial Risk Management with Python Abdullah Karasan
# - [FF3 dataset](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
#
# </font>
# </div>

# In[ ]:
