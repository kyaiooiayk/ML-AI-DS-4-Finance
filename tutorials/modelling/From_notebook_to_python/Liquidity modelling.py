#!/usr/bin/env python
# coding: utf-8

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Liquidity modelling
# 
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import statsmodels.api as sm
warnings.filterwarnings("ignore")
plt.rcParams['figure.figsize'] = (10, 6)
pd.set_option('use_inf_as_na', True)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


# # Import data
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - ask (`ASKHI`)
# - bid (`BIDLO`)
# - open (`OPENPRC`)
# - trading price (`PRC`) 
# - volume (`VOL`)
# - return (`RET`)
# - volume-weighted return (`vwretx`) of the stock
# - number of shares outstanding (`SHROUT`):
# 
# </font>
# </div>

# In[2]:


liq_data = pd.read_csv('./bid_ask.csv')


# In[3]:


liq_data.head()


# # Pre-processing
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - Calculating some liquidity measures requires a rolling-window estimation, such as the calculation of the bid price for five days. 
# - To accomplish this task, the list named `rolling_five` is generated using the following code.
# 
# </font>
# </div>

# In[4]:


rolling_five = []

for j in liq_data.TICKER.unique():
    for i in range(len(liq_data[liq_data.TICKER == j])):
        rolling_five.append(liq_data[i:i+5].agg({'BIDLO': 'min',
                                                'ASKHI': 'max',
                                                 'VOL': 'sum',
                                                 'SHROUT': 'mean',
                                                 'PRC': 'mean'}))


# In[5]:


rolling_five_df = pd.DataFrame(rolling_five)
rolling_five_df.columns = ['bidlo_min', 'askhi_max', 'vol_sum',
                           'shrout_mean', 'prc_mean']
liq_vol_all = pd.concat([liq_data, rolling_five_df], axis=1)


# In[6]:


liq_vol_all


# # Liquidity Ratio
# <hr style = "border:2px solid black" ></hr>

# In[7]:


liq_ratio = []

for j in liq_vol_all.TICKER.unique():
    for i in range(len(liq_vol_all[liq_vol_all.TICKER == j])):
        liq_ratio.append((liq_vol_all['PRC'][i+1:i+6] * 
                          liq_vol_all['VOL'][i+1:i+6]).sum()/
                         (np.abs(liq_vol_all['PRC'][i+1:i+6].mean() - 
                                 liq_vol_all['PRC'][i:i+5].mean())))


# In[8]:


liq_ratio 


# # Hui-Heubel liquidity ratio
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# 
# </font>
# </div>

# In[9]:


Lhh = []

for j in liq_vol_all.TICKER.unique():
    for i in range(len(liq_vol_all[liq_vol_all.TICKER == j])):
        Lhh.append((liq_vol_all['PRC'][i:i+5].max() - 
                    liq_vol_all['PRC'][i:i+5].min()) /  
                   liq_vol_all['PRC'][i:i+5].min() /  
                   (liq_vol_all['VOL'][i:i+5].sum() / 
                    liq_vol_all['SHROUT'][i:i+5].mean() * 
                    liq_vol_all['PRC'][i:i+5].mean()))


# # Turnover ratio
# <hr style = "border:2px solid black" ></hr>

# In[10]:


turnover_ratio = []

for j in liq_vol_all.TICKER.unique():
    for i in range(len(liq_vol_all[liq_vol_all.TICKER == j])):
        turnover_ratio.append((1/liq_vol_all['VOL'].count()) * 
                              (np.sum(liq_vol_all['VOL'][i:i+1]) / 
                               np.sum(liq_vol_all['SHROUT'][i:i+1])))


# In[11]:


liq_vol_all['liq_ratio'] = pd.DataFrame(liq_ratio)
liq_vol_all['Lhh'] = pd.DataFrame(Lhh)
liq_vol_all['turnover_ratio'] = pd.DataFrame(turnover_ratio)


# # Transaction Cost-base liquidity measures
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# 
# - Transaction cost is a cost an investor must bear during trade. It is referred to as any expenses related to the execution of trade. It can be splitted in two:
#     - **Explicit cost** relates to order processing, taxes, and brokerage fees
#     - **Implicit cost** includes more latent costs, such as bid-ask spread, timing of execution, and so on.
# 
# - Transaction cost is related to the tightness and immediacy dimensions of liquidity. 
#     
# - High transaction costs discourage investors to trade and this, in turn, decreases the number of buyers and sellers in the market so that the trading place diverges away from the more centralized market into a fragmented one, which result in a shallow market. To the extent that transaction cost is low, investors are
# willing to trade and this results in a flourished trading environment in which markets will be more centralized.
# 
# - Similarly, an abundance of buyers and sellers in a low transaction cost environment refers to the fact that a large number of orders are traded in a short period of time. So, immediacy is the other dimension of liquidity, which is closely related to the transaction cost.
#     
# </font>
# </div>

# ### Bid-Ask Spreads

# <div class="alert alert-info">
# <font color=black>
#     
# - Even though there is an ongoing debate about the goodness of bid-ask spread as well as the assurance that these models provide, bid-ask spread is a widely recognized proxy for transaction cost. 
# 
# - To the extent that bid-ask spread is a good analysis of transaction cost, it is also a good indicator of liquidity by which the ease of converting an asset into cash (or a cash equivalent) might be determined. 
#     
# - The other two well-known bid-ask spreads are:
#     - percentage quoted
#     - percentage effective bid-ask spreads
# 
# </font>
# </div>

# In[12]:


liq_vol_all['mid_price'] = (liq_vol_all.ASKHI + liq_vol_all.BIDLO) / 2
liq_vol_all['percent_quoted_ba'] = (liq_vol_all.ASKHI - 
                                    liq_vol_all.BIDLO) / \
                                    liq_vol_all.mid_price
liq_vol_all['percent_effective_ba'] = 2 * abs((liq_vol_all.PRC - 
                                               liq_vol_all.mid_price)) / \
                                               liq_vol_all.mid_price


# In[13]:


liq_vol_all[['mid_price', 'percent_quoted_ba', 'percent_effective_ba']]


# ### Roll's Spread

# <div class="alert alert-info">
# <font color=black>
# 
# - Assuming that the market is efficient and the probability of distribution of observed price changes is stationary, Roll’s spread **is motivated by** the fact that serial correlation of price changes is a good proxy for liquidity.
# 
# - One of the most important things to note in calculating Roll’s spread is that positive covariance is not well-defined, and it consists of almost half of the cases. The literature puts forth several methods to remedy this shortcoming, and we’ll embrace one of this.
# 
# </font>
# </div>

# In[14]:


liq_vol_all['price_diff'] = liq_vol_all.groupby('TICKER')['PRC']\
                            .apply(lambda x:x.diff())
liq_vol_all.dropna(inplace=True)
roll = []

for j in liq_vol_all.TICKER.unique():
     for i in range(len(liq_vol_all[liq_vol_all.TICKER == j])):
        roll_cov = np.cov(liq_vol_all['price_diff'][i:i+5], 
                          liq_vol_all['price_diff'][i+1:i+6])
        if roll_cov[0,1] < 0:
            roll.append(2 * np.sqrt(-roll_cov[0, 1]))
        else:
             roll.append(2 * np.sqrt(np.abs(roll_cov[0, 1])))


# ### Corwin and Schultz (2012)

# <div class="alert alert-info">
# <font color=black>
# 
# - The Corwin-Schultz spread is rather intuitive and easy to apply. 
# - It rests mainly on the following assumption: given that the daily high and low prices are typically buyer and seller initiated, respectively, the observed price change can be split into effective price volatility and bid-ask spread. 
# - So the ratio of high-to-low prices for a day reflects both the stock’s variance and its bid-ask spread
#     
# </font>
# </div>

# In[15]:


gamma = []

for j in liq_vol_all.TICKER.unique():
    for i in range(len(liq_vol_all[liq_vol_all.TICKER == j])):
        gamma.append((max(liq_vol_all['ASKHI'].iloc[i+1], 
                          liq_vol_all['ASKHI'].iloc[i]) - 
                      min(liq_vol_all['BIDLO'].iloc[i+1], 
                          liq_vol_all['BIDLO'].iloc[i])) ** 2)
        gamma_array = np.array(gamma)


# In[16]:


beta = []

for j in liq_vol_all.TICKER.unique():
    for i in range(len(liq_vol_all[liq_vol_all.TICKER == j])):
        beta.append((liq_vol_all['ASKHI'].iloc[i+1] - 
                     liq_vol_all['BIDLO'].iloc[i+1]) ** 2 + 
                    (liq_vol_all['ASKHI'].iloc[i] - 
                     liq_vol_all['BIDLO'].iloc[i]) ** 2)
        beta_array = np.array(beta)


# In[17]:


alpha = ((np.sqrt(2 * beta_array) - np.sqrt(beta_array)) / 
       (3 - (2 * np.sqrt(2)))) - np.sqrt(gamma_array / 
                                         (3 - (2 * np.sqrt(2))))
CS_spread = (2 * np.exp(alpha - 1)) / (1 + np.exp(alpha))


# In[18]:


liq_vol_all = liq_vol_all.reset_index()
liq_vol_all['roll'] = pd.DataFrame(roll)
liq_vol_all['CS_spread'] = pd.DataFrame(CS_spread)


# # Price Based Measures
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - Price impact–based liquidity measures by which we are able to gauge the extent to which price is sensitive to volume and turnover ratio. Recall that resiliency refers to the market responsiveness about new orders. 
# - If the market is responsive to the new order—that is, a new order correct the imbalances in the market—then it is said to be resilient. 
# - Thus, given a change in volume and/or turnover ratio, high price adjustment amounts to resiliency or vice versa.
# 
# </font>
# </div>

# ## Amihud illiquidity measure

# In[19]:


dvol = []

for j in liq_vol_all.TICKER.unique():
    for i in range(len(liq_vol_all[liq_vol_all.TICKER == j])):
        dvol.append((liq_vol_all['PRC'][i:i+5] *
                     liq_vol_all['VOL'][i:i+5]).sum())
liq_vol_all['dvol'] = pd.DataFrame(dvol)


# In[20]:


amihud = []

for j in liq_vol_all.TICKER.unique():
    for i in range(len(liq_vol_all[liq_vol_all.TICKER == j])):
        amihud.append((1 / liq_vol_all['RET'].count()) * 
                      (np.sum(np.abs(liq_vol_all['RET'][i:i+1])) / 
                              np.sum(liq_vol_all['dvol'][i:i+1])))
liq_vol_all['amihud'] = pd.DataFrame(amihud)


# ## Florackis, Andros, and Alexandros (2011) price impact ratio

# In[21]:


florackis = []

for j in liq_vol_all.TICKER.unique():
    for i in range(len(liq_vol_all[liq_vol_all.TICKER == j])):
        florackis.append((1 / liq_vol_all['RET'].count()) * 
                         (np.sum(np.abs(liq_vol_all['RET'][i:i+1]) / 
                                 liq_vol_all['turnover_ratio'][i:i+1])))
    
liq_vol_all['florackis'] = pd.DataFrame(florackis)


# ## Coefficient of elasticity of trading (CET)

# In[22]:


liq_vol_all['vol_diff_pct'] = liq_vol_all.groupby('TICKER')['VOL']\
                              .apply(lambda x: x.diff()).pct_change()
liq_vol_all['price_diff_pct'] = liq_vol_all.groupby('TICKER')['PRC']\
                              .apply(lambda x: x.diff()).pct_change()


# In[23]:


cet = []

for j in liq_vol_all.TICKER.unique():
    for i in range(len(liq_vol_all[liq_vol_all.TICKER == j])):
        cet.append(np.sum(liq_vol_all['vol_diff_pct'][i:i+1])/
                   np.sum(liq_vol_all['price_diff_pct'][i:i+1]))

liq_vol_all['cet'] = pd.DataFrame(cet)


# In[24]:


liq_vol_all[['amihud', "florackis", "cet"]]


# # Market Impact-based Measures
# <hr style = "border:2px solid black" ></hr>

# In[26]:


liq_vol_all['VOL_pct_change'] = liq_vol_all.groupby('TICKER')['VOL']\
                                .apply(lambda x: x.pct_change())
liq_vol_all.dropna(subset=['VOL_pct_change'], inplace=True)
liq_vol_all = liq_vol_all.reset_index()


# In[27]:


unsys_resid = []

for i in liq_vol_all.TICKER.unique():
    X1 = liq_vol_all[liq_vol_all['TICKER'] == i]['vwretx']
    y = liq_vol_all[liq_vol_all['TICKER'] == i]['RET']
    ols = sm.OLS(y, X1).fit()
    unsys_resid.append(ols.resid)


# In[28]:


market_impact = {}

for i, j in zip(liq_vol_all.TICKER.unique(), 
                range(len(liq_vol_all['TICKER'].unique()))):
    X2 = liq_vol_all[liq_vol_all['TICKER'] == i]['VOL_pct_change']
    ols = sm.OLS(unsys_resid[j] ** 2, X2).fit()
    print('***' * 30)
    print(f'OLS Result for {i}')
    print(ols.summary())
    market_impact[j] = ols.resid


# In[29]:


append1 = market_impact[0].append(market_impact[1])
liq_vol_all['market_impact'] = append1.append(market_impact[2])


# In[30]:


cols = ['vol_diff_pct', 'price_diff_pct', 'price_diff',
        'VOL_pct_change', 'dvol', 'mid_price']
liq_measures_all = liq_vol_all.drop(liq_vol_all[cols], axis=1)\
                   .iloc[:, -11:]
liq_measures_all.dropna(inplace=True)
liq_measures_all.describe().T


# ## GMM

# In[31]:


from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# In[32]:


liq_measures_all2 = liq_measures_all.dropna()
scaled_liq = StandardScaler().fit_transform(liq_measures_all2)


# In[33]:


kwargs = dict(alpha=0.5, bins=50,  stacked=True)
plt.hist(liq_measures_all.loc[:, 'percent_quoted_ba'],
         **kwargs, label='TC-based')
plt.hist(liq_measures_all.loc[:, 'turnover_ratio'],
         **kwargs, label='Volume-based')
plt.hist(liq_measures_all.loc[:, 'market_impact'],
         **kwargs, label='Market-based')
plt.title('Multimodality of the Liquidity Measures')
plt.legend()
plt.show()


# In[34]:


n_components = np.arange(1, 10)
clusters = [GaussianMixture(n, covariance_type='spherical',
                            random_state=0).fit(scaled_liq)
          for n in n_components]
plt.plot(n_components, [m.bic(scaled_liq) for m in clusters])
plt.title('Optimum Number of Components')
plt.xlabel('n_components')
plt.ylabel('BIC values')
plt.show()


# In[35]:


def cluster_state(data, nstates):
    gmm = GaussianMixture(n_components=nstates,
                          covariance_type='spherical',
                          init_params='kmeans')
    gmm_fit = gmm.fit(scaled_liq)
    labels = gmm_fit.predict(scaled_liq)
    state_probs = gmm.predict_proba(scaled_liq)
    state_probs_df = pd.DataFrame(state_probs,
                                  columns=['state-1', 'state-2', 'state-3'])
    state_prob_means = [state_probs_df.iloc[:, i].mean()
                        for i in range(len(state_probs_df.columns))]
    if np.max(state_prob_means) == state_prob_means[0]:
        print('State-1 is likely to occur with a probability of {:4f}'
              .format(state_prob_means[0]))
    elif np.max(state_prob_means) == state_prob_means[1]:
        print('State-2 is likely to occur with a probability of {:4f}'
              .format(state_prob_means[1]))
    else:
        print('State-3 is likely to occur with a probability of {:4f}'
              .format(state_prob_means[2]))
    return state_probs


# In[36]:


state_probs = cluster_state(scaled_liq, 3)
print(f'State probabilities are {state_probs.mean(axis=0)}')


# In[37]:


from sklearn.decomposition import PCA


# In[38]:


pca = PCA(n_components=11)
components = pca.fit_transform(scaled_liq)
plt.plot(pca.explained_variance_ratio_)
plt.title('Scree Plot')
plt.xlabel('Number of Components')
plt.ylabel('% of Explained Variance')
plt.show()


# In[39]:


def gmm_pca(data, nstate):
    pca = PCA(n_components=3)
    components = pca.fit_transform(data)
    mxtd = GaussianMixture(n_components=nstate,
                           covariance_type='spherical')
    gmm = mxtd.fit(components)
    labels = gmm.predict(components)
    state_probs = gmm.predict_proba(components)
    return state_probs, pca


# In[40]:


state_probs, pca = gmm_pca(scaled_liq, 3)
print(f'State probabilities are {state_probs.mean(axis=0)}')


# In[41]:


def wpc():
    state_probs_df = pd.DataFrame(state_probs,
                                  columns=['state-1', 'state-2',
                                           'state-3'])
    state_prob_means = [state_probs_df.iloc[:, i].mean() 
                        for i in range(len(state_probs_df.columns))]
    if np.max(state_prob_means) == state_prob_means[0]:
        print('State-1 is likely to occur with a probability of {:4f}'
              .format(state_prob_means[0]))
    elif np.max(state_prob_means) == state_prob_means[1]:
        print('State-2 is likely to occur with a probability of {:4f}'
              .format(state_prob_means[1]))
    else:
        print('State-3 is likely to occur with a probability of {:4f}'
              .format(state_prob_means[2]))
wpc()


# In[42]:


loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loading_matrix = pd.DataFrame(loadings, 
                              columns=['PC1', 'PC2', 'PC3'],
                              index=liq_measures_all.columns)
loading_matrix


# ## GMCM

# <div class="alert alert-info">
# <font color=black>
# 
# - Copula is a function that maps marginal distribution of individual risks to multivariate distribution, resulting in a joint distribution of many standard uniform random variables. 
# 
# - If we are working with a known distribution, such as normal distribution, it is easy to model joint distribution of variables, known as bivariate normal. However, the challenge here is to define the correlation structure between these two variables, and this is the point at which copulas come in
# 
# </font>
# </div>

# In[43]:


from copulae.mixtures.gmc.gmc import GaussianMixtureCopula


# In[44]:


_, dim = scaled_liq.shape
gmcm = GaussianMixtureCopula(n_clusters=3, ndim=dim)


# In[45]:


gmcm_fit = gmcm.fit(scaled_liq, method='kmeans',
                    criteria='GMCM', eps=0.0001)
state_prob = gmcm_fit.params.prob
print(f'The state {np.argmax(state_prob) + 1} is likely to occur')
print(f'State probabilities based on GMCM are {state_prob}')


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://github.com/abdullahkarasan/mlfrm/blob/main/codes/chp_7.ipynb
# - Machine Learning for Financial Risk Management with Python Abdullah Karasan
# 
# </font>
# </div>
#     
