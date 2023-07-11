#!/usr/bin/env python
# coding: utf-8

# # Stochastics

# In[1]:


import math
import numpy as np
import numpy.random as npr
from pylab import plt, mpl


# In[2]:


plt.style.use("seaborn")
mpl.rcParams["font.family"] = "serif"
get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'svg'")


# ## Random Numbers

# In[3]:


npr.seed(100)
np.set_printoptions(precision=4)


# In[4]:


npr.rand(10)


# In[5]:


npr.rand(5, 5)


# In[6]:


a = 5.0
b = 10.0
npr.rand(10) * (b - a) + a


# In[7]:


npr.rand(5, 5) * (b - a) + a


# In[8]:


sample_size = 500
rn1 = npr.rand(sample_size, 3)
rn2 = npr.randint(0, 10, sample_size)
rn3 = npr.sample(size=sample_size)
a = [0, 25, 50, 75, 100]
rn4 = npr.choice(a, size=sample_size)


# In[9]:


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
ax1.hist(rn1, bins=25, stacked=True)
ax1.set_title("rand")
ax1.set_ylabel("frequency")
ax2.hist(rn2, bins=25)
ax2.set_title("randint")
ax3.hist(rn3, bins=25)
ax3.set_title("sample")
ax3.set_ylabel("frequency")
ax4.hist(rn4, bins=25)
ax4.set_title("choice")
# plt.savefig('../../images/ch12/stoch_01.png');


# In[10]:


sample_size = 500
rn1 = npr.standard_normal(sample_size)
rn2 = npr.normal(100, 20, sample_size)
rn3 = npr.chisquare(df=0.5, size=sample_size)
rn4 = npr.poisson(lam=1.0, size=sample_size)


# In[11]:


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
ax1.hist(rn1, bins=25)
ax1.set_title("standard normal")
ax1.set_ylabel("frequency")
ax2.hist(rn2, bins=25)
ax2.set_title("normal(100, 20)")
ax3.hist(rn3, bins=25)
ax3.set_title("chi square")
ax3.set_ylabel("frequency")
ax4.hist(rn4, bins=25)
ax4.set_title("Poisson")
# plt.savefig('../../images/ch12/stoch_02.png');


# ## Simulation

# ### Random Variables

# In[12]:


S0 = 100
r = 0.05
sigma = 0.25
T = 2.0
I = 10000
ST1 = S0 * np.exp(
    (r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * npr.standard_normal(I)
)


# In[13]:


plt.figure(figsize=(10, 6))
plt.hist(ST1, bins=50)
plt.xlabel("index level")
plt.ylabel("frequency")
# plt.savefig('../../images/ch12/stoch_03.png');


# In[14]:


ST2 = S0 * npr.lognormal((r - 0.5 * sigma**2) * T, sigma * math.sqrt(T), size=I)


# In[15]:


plt.figure(figsize=(10, 6))
plt.hist(ST2, bins=50)
plt.xlabel("index level")
plt.ylabel("frequency")
# plt.savefig('../../images/ch12/stoch_04.png');


# In[16]:


import scipy.stats as scs


# In[17]:


def print_statistics(a1, a2):
    """Prints selected statistics.

    Parameters
    ==========
    a1, a2: ndarray objects
        results objects from simulation
    """
    sta1 = scs.describe(a1)
    sta2 = scs.describe(a2)
    print("%14s %14s %14s" % ("statistic", "data set 1", "data set 2"))
    print(45 * "-")
    print("%14s %14.3f %14.3f" % ("size", sta1[0], sta2[0]))
    print("%14s %14.3f %14.3f" % ("min", sta1[1][0], sta2[1][0]))
    print("%14s %14.3f %14.3f" % ("max", sta1[1][1], sta2[1][1]))
    print("%14s %14.3f %14.3f" % ("mean", sta1[2], sta2[2]))
    print("%14s %14.3f %14.3f" % ("std", np.sqrt(sta1[3]), np.sqrt(sta2[3])))
    print("%14s %14.3f %14.3f" % ("skew", sta1[4], sta2[4]))
    print("%14s %14.3f %14.3f" % ("kurtosis", sta1[5], sta2[5]))


# In[18]:


print_statistics(ST1, ST2)


# ### Stochastic Processes

# #### Geometric Brownian Motion

# In[19]:


I = 10000
M = 50
dt = T / M
S = np.zeros((M + 1, I))
S[0] = S0
for t in range(1, M + 1):
    S[t] = S[t - 1] * np.exp(
        (r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * npr.standard_normal(I)
    )


# In[20]:


plt.figure(figsize=(10, 6))
plt.hist(S[-1], bins=50)
plt.xlabel("index level")
plt.ylabel("frequency")
# plt.savefig('../../images/ch12/stoch_05.png');


# In[21]:


print_statistics(S[-1], ST2)


# In[22]:


plt.figure(figsize=(10, 6))
plt.plot(S[:, :10], lw=1.5)
plt.xlabel("time")
plt.ylabel("index level")
# plt.savefig('../../images/ch12/stoch_06.png');


# #### Square-Root Diffusion

# In[23]:


x0 = 0.05
kappa = 3.0
theta = 0.02
sigma = 0.1
I = 10000
M = 50
dt = T / M


# In[24]:


def srd_euler():
    xh = np.zeros((M + 1, I))
    x = np.zeros_like(xh)
    xh[0] = x0
    x[0] = x0
    for t in range(1, M + 1):
        xh[t] = (
            xh[t - 1]
            + kappa * (theta - np.maximum(xh[t - 1], 0)) * dt
            + sigma
            * np.sqrt(np.maximum(xh[t - 1], 0))
            * math.sqrt(dt)
            * npr.standard_normal(I)
        )
    x = np.maximum(xh, 0)
    return x


x1 = srd_euler()


# In[25]:


plt.figure(figsize=(10, 6))
plt.hist(x1[-1], bins=50)
plt.xlabel("value")
plt.ylabel("frequency")
# plt.savefig('../../images/ch12/stoch_07.png');


# In[26]:


plt.figure(figsize=(10, 6))
plt.plot(x1[:, :10], lw=1.5)
plt.xlabel("time")
plt.ylabel("index level")
# plt.savefig('../../images/ch12/stoch_08.png');


# In[27]:


def srd_exact():
    x = np.zeros((M + 1, I))
    x[0] = x0
    for t in range(1, M + 1):
        df = 4 * theta * kappa / sigma**2
        c = (sigma**2 * (1 - np.exp(-kappa * dt))) / (4 * kappa)
        nc = np.exp(-kappa * dt) / c * x[t - 1]
        x[t] = c * npr.noncentral_chisquare(df, nc, size=I)
    return x


x2 = srd_exact()


# In[28]:


plt.figure(figsize=(10, 6))
plt.hist(x2[-1], bins=50)
plt.xlabel("value")
plt.ylabel("frequency")
# plt.savefig('../../images/ch12/stoch_09.png');


# In[29]:


plt.figure(figsize=(10, 6))
plt.plot(x2[:, :10], lw=1.5)
plt.xlabel("time")
plt.ylabel("index level")
# plt.savefig('../../images/ch12/stoch_10.png');


# In[30]:


print_statistics(x1[-1], x2[-1])


# In[31]:


I = 250000
get_ipython().run_line_magic("time", "x1 = srd_euler()")


# In[32]:


get_ipython().run_line_magic("time", "x2 = srd_exact()")


# In[33]:


print_statistics(x1[-1], x2[-1])
x1 = 0.0
x2 = 0.0


# #### Stochastic Volatility

# In[34]:


S0 = 100.0
r = 0.05
v0 = 0.1
kappa = 3.0
theta = 0.25
sigma = 0.1
rho = 0.6
T = 1.0


# In[35]:


corr_mat = np.zeros((2, 2))
corr_mat[0, :] = [1.0, rho]
corr_mat[1, :] = [rho, 1.0]
cho_mat = np.linalg.cholesky(corr_mat)


# In[36]:


cho_mat


# In[37]:


M = 50
I = 10000
dt = T / M


# In[38]:


ran_num = npr.standard_normal((2, M + 1, I))


# In[39]:


v = np.zeros_like(ran_num[0])
vh = np.zeros_like(v)


# In[40]:


v[0] = v0
vh[0] = v0


# In[41]:


for t in range(1, M + 1):
    ran = np.dot(cho_mat, ran_num[:, t, :])
    vh[t] = (
        vh[t - 1]
        + kappa * (theta - np.maximum(vh[t - 1], 0)) * dt
        + sigma * np.sqrt(np.maximum(vh[t - 1], 0)) * math.sqrt(dt) * ran[1]
    )


# In[42]:


v = np.maximum(vh, 0)


# In[43]:


S = np.zeros_like(ran_num[0])
S[0] = S0
for t in range(1, M + 1):
    ran = np.dot(cho_mat, ran_num[:, t, :])
    S[t] = S[t - 1] * np.exp(
        (r - 0.5 * v[t]) * dt + np.sqrt(v[t]) * ran[0] * np.sqrt(dt)
    )


# In[44]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
ax1.hist(S[-1], bins=50)
ax1.set_xlabel("index level")
ax1.set_ylabel("frequency")
ax2.hist(v[-1], bins=50)
ax2.set_xlabel("volatility")
# plt.savefig('../../images/ch12/stoch_11.png');


# In[45]:


print_statistics(S[-1], v[-1])


# In[46]:


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
ax1.plot(S[:, :10], lw=1.5)
ax1.set_ylabel("index level")
ax2.plot(v[:, :10], lw=1.5)
ax2.set_xlabel("time")
ax2.set_ylabel("volatility")
# plt.savefig('../../images/ch12/stoch_12.png');


# #### Jump-Diffusion

# In[47]:


S0 = 100.0
r = 0.05
sigma = 0.2
lamb = 0.75
mu = -0.6
delta = 0.25
rj = lamb * (math.exp(mu + 0.5 * delta**2) - 1)


# In[48]:


T = 1.0
M = 50
I = 10000
dt = T / M


# In[49]:


S = np.zeros((M + 1, I))
S[0] = S0
sn1 = npr.standard_normal((M + 1, I))
sn2 = npr.standard_normal((M + 1, I))
poi = npr.poisson(lamb * dt, (M + 1, I))
for t in range(1, M + 1, 1):
    S[t] = S[t - 1] * (
        np.exp((r - rj - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * sn1[t])
        + (np.exp(mu + delta * sn2[t]) - 1) * poi[t]
    )
    S[t] = np.maximum(S[t], 0)


# In[50]:


plt.figure(figsize=(10, 6))
plt.hist(S[-1], bins=50)
plt.xlabel("value")
plt.ylabel("frequency")
# plt.savefig('../../images/ch12/stoch_13.png');


# In[51]:


plt.figure(figsize=(10, 6))
plt.plot(S[:, :10], lw=1.5)
plt.xlabel("time")
plt.ylabel("index level")
# plt.savefig('../../images/ch12/stoch_14.png');


# ### Variance Reduction

# In[52]:


print("%15s %15s" % ("Mean", "Std. Deviation"))
print(31 * "-")
for i in range(1, 31, 2):
    npr.seed(100)
    sn = npr.standard_normal(i**2 * 10000)
    print("%15.12f %15.12f" % (sn.mean(), sn.std()))


# In[53]:


i**2 * 10000


# In[54]:


sn = npr.standard_normal(int(10000 / 2))
sn = np.concatenate((sn, -sn))


# In[55]:


np.shape(sn)


# In[56]:


sn.mean()


# In[57]:


print("%15s %15s" % ("Mean", "Std. Deviation"))
print(31 * "-")
for i in range(1, 31, 2):
    npr.seed(1000)
    sn = npr.standard_normal(i**2 * int(10000 / 2))
    sn = np.concatenate((sn, -sn))
    print("%15.12f %15.12f" % (sn.mean(), sn.std()))


# In[58]:


sn = npr.standard_normal(10000)


# In[59]:


sn.mean()


# In[60]:


sn.std()


# In[61]:


sn_new = (sn - sn.mean()) / sn.std()


# In[62]:


sn_new.mean()


# In[63]:


sn_new.std()


# In[64]:


def gen_sn(M, I, anti_paths=True, mo_match=True):
    """Function to generate random numbers for simulation.

    Parameters
    ==========
    M: int
        number of time intervals for discretization
    I: int
        number of paths to be simulated
    anti_paths: boolean
        use of antithetic variates
    mo_math: boolean
        use of moment matching
    """
    if anti_paths is True:
        sn = npr.standard_normal((M + 1, int(I / 2)))
        sn = np.concatenate((sn, -sn), axis=1)
    else:
        sn = npr.standard_normal((M + 1, I))
    if mo_match is True:
        sn = (sn - sn.mean()) / sn.std()
    return sn


# ## Valuation

# ### European Options

# In[65]:


S0 = 100.0
r = 0.05
sigma = 0.25
T = 1.0
I = 50000


# In[66]:


def gbm_mcs_stat(K):
    """Valuation of European call option in Black-Scholes-Merton
    by Monte Carlo simulation (of index level at maturity)

    Parameters
    ==========
    K: float
        (positive) strike price of the option

    Returns
    =======
    C0: float
        estimated present value of European call option
    """
    sn = gen_sn(1, I)
    # simulate index level at maturity
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * sn[1])
    # calculate payoff at maturity
    hT = np.maximum(ST - K, 0)
    # calculate MCS estimator
    C0 = math.exp(-r * T) * np.mean(hT)
    return C0


# In[67]:


gbm_mcs_stat(K=105.0)


# In[68]:


M = 50


# In[69]:


def gbm_mcs_dyna(K, option="call"):
    """Valuation of European options in Black-Scholes-Merton
    by Monte Carlo simulation (of index level paths)

    Parameters
    ==========
    K: float
        (positive) strike price of the option
    option : string
        type of the option to be valued ('call', 'put')

    Returns
    =======
    C0: float
        estimated present value of European call option
    """
    dt = T / M
    # simulation of index level paths
    S = np.zeros((M + 1, I))
    S[0] = S0
    sn = gen_sn(M, I)
    for t in range(1, M + 1):
        S[t] = S[t - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * sn[t]
        )
    # case-based calculation of payoff
    if option == "call":
        hT = np.maximum(S[-1] - K, 0)
    else:
        hT = np.maximum(K - S[-1], 0)
    # calculation of MCS estimator
    C0 = math.exp(-r * T) * np.mean(hT)
    return C0


# In[70]:


gbm_mcs_dyna(K=110.0, option="call")


# In[71]:


gbm_mcs_dyna(K=110.0, option="put")


# In[72]:


from bsm_functions import bsm_call_value


# In[73]:


stat_res = []
dyna_res = []
anal_res = []
k_list = np.arange(80.0, 120.1, 5.0)
np.random.seed(100)


# In[74]:


for K in k_list:
    stat_res.append(gbm_mcs_stat(K))
    dyna_res.append(gbm_mcs_dyna(K))
    anal_res.append(bsm_call_value(S0, K, T, r, sigma))


# In[75]:


stat_res = np.array(stat_res)
dyna_res = np.array(dyna_res)
anal_res = np.array(anal_res)


# In[76]:


plt.figure(figsize=(10, 6))
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
ax1.plot(k_list, anal_res, "b", label="analytical")
ax1.plot(k_list, stat_res, "ro", label="static")
ax1.set_ylabel("European call option value")
ax1.legend(loc=0)
ax1.set_ylim(bottom=0)
wi = 1.0
ax2.bar(k_list - wi / 2, (anal_res - stat_res) / anal_res * 100, wi)
ax2.set_xlabel("strike")
ax2.set_ylabel("difference in %")
ax2.set_xlim(left=75, right=125)
# plt.savefig('../../images/ch12/stoch_15.png');


# In[77]:


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
ax1.plot(k_list, anal_res, "b", label="analytical")
ax1.plot(k_list, dyna_res, "ro", label="dynamic")
ax1.set_ylabel("European call option value")
ax1.legend(loc=0)
ax1.set_ylim(bottom=0)
wi = 1.0
ax2.bar(k_list - wi / 2, (anal_res - dyna_res) / anal_res * 100, wi)
ax2.set_xlabel("strike")
ax2.set_ylabel("difference in %")
ax2.set_xlim(left=75, right=125)
# plt.savefig('../../images/ch12/stoch_16.png');


# ### American Options

# In[78]:


def gbm_mcs_amer(K, option="call"):
    """Valuation of American option in Black-Scholes-Merton
    by Monte Carlo simulation by LSM algorithm

    Parameters
    ==========
    K : float
        (positive) strike price of the option
    option : string
        type of the option to be valued ('call', 'put')

    Returns
    =======
    C0 : float
        estimated present value of European call option
    """
    dt = T / M
    df = math.exp(-r * dt)
    # simulation of index levels
    S = np.zeros((M + 1, I))
    S[0] = S0
    sn = gen_sn(M, I)
    for t in range(1, M + 1):
        S[t] = S[t - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * sn[t]
        )
    # case based calculation of payoff
    if option == "call":
        h = np.maximum(S - K, 0)
    else:
        h = np.maximum(K - S, 0)
    # LSM algorithm
    V = np.copy(h)
    for t in range(M - 1, 0, -1):
        reg = np.polyfit(S[t], V[t + 1] * df, 7)
        C = np.polyval(reg, S[t])
        V[t] = np.where(C > h[t], V[t + 1] * df, h[t])
    # MCS estimator
    C0 = df * np.mean(V[1])
    return C0


# In[79]:


gbm_mcs_amer(110.0, option="call")


# In[80]:


gbm_mcs_amer(110.0, option="put")


# In[81]:


euro_res = []
amer_res = []


# In[82]:


k_list = np.arange(80.0, 120.1, 5.0)


# In[83]:


for K in k_list:
    euro_res.append(gbm_mcs_dyna(K, "put"))
    amer_res.append(gbm_mcs_amer(K, "put"))


# In[84]:


euro_res = np.array(euro_res)
amer_res = np.array(amer_res)


# In[85]:


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
ax1.plot(k_list, euro_res, "b", label="European put")
ax1.plot(k_list, amer_res, "ro", label="American put")
ax1.set_ylabel("call option value")
ax1.legend(loc=0)
wi = 1.0
ax2.bar(k_list - wi / 2, (amer_res - euro_res) / euro_res * 100, wi)
ax2.set_xlabel("strike")
ax2.set_ylabel("early exercise premium in %")
ax2.set_xlim(left=75, right=125)
# plt.savefig('../../images/ch12/stoch_17.png');


# ## Risk Measures

# ### Value-at-Risk

# In[86]:


S0 = 100
r = 0.05
sigma = 0.25
T = 30 / 365.0
I = 10000


# In[87]:


ST = S0 * np.exp(
    (r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * npr.standard_normal(I)
)


# In[88]:


R_gbm = np.sort(ST - S0)


# In[89]:


plt.figure(figsize=(10, 6))
plt.hist(R_gbm, bins=50)
plt.xlabel("absolute return")
plt.ylabel("frequency")
# plt.savefig('../../images/ch12/stoch_18.png');


# In[90]:


import warnings

warnings.simplefilter("ignore")


# In[91]:


percs = [0.01, 0.1, 1.0, 2.5, 5.0, 10.0]
var = scs.scoreatpercentile(R_gbm, percs)
print("%16s %16s" % ("Confidence Level", "Value-at-Risk"))
print(33 * "-")
for pair in zip(percs, var):
    print("%16.2f %16.3f" % (100 - pair[0], -pair[1]))


# In[92]:


dt = 30.0 / 365 / M
rj = lamb * (math.exp(mu + 0.5 * delta**2) - 1)


# In[93]:


S = np.zeros((M + 1, I))
S[0] = S0
sn1 = npr.standard_normal((M + 1, I))
sn2 = npr.standard_normal((M + 1, I))
poi = npr.poisson(lamb * dt, (M + 1, I))
for t in range(1, M + 1, 1):
    S[t] = S[t - 1] * (
        np.exp((r - rj - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * sn1[t])
        + (np.exp(mu + delta * sn2[t]) - 1) * poi[t]
    )
    S[t] = np.maximum(S[t], 0)


# In[94]:


R_jd = np.sort(S[-1] - S0)


# In[95]:


plt.figure(figsize=(10, 6))
plt.hist(R_jd, bins=50)
plt.xlabel("absolute return")
plt.ylabel("frequency")
# plt.savefig('../../images/ch12/stoch_19.png');


# In[96]:


percs = [0.01, 0.1, 1.0, 2.5, 5.0, 10.0]
var = scs.scoreatpercentile(R_jd, percs)
print("%16s %16s" % ("Confidence Level", "Value-at-Risk"))
print(33 * "-")
for pair in zip(percs, var):
    print("%16.2f %16.3f" % (100 - pair[0], -pair[1]))


# In[97]:


percs = list(np.arange(0.0, 10.1, 0.1))
gbm_var = scs.scoreatpercentile(R_gbm, percs)
jd_var = scs.scoreatpercentile(R_jd, percs)


# In[98]:


plt.figure(figsize=(10, 6))
plt.plot(percs, gbm_var, "b", lw=1.5, label="GBM")
plt.plot(percs, jd_var, "r", lw=1.5, label="JD")
plt.legend(loc=4)
plt.xlabel("100 - confidence level [%]")
plt.ylabel("value-at-risk")
plt.ylim(ymax=0.0)
# plt.savefig('../../images/ch12/stoch_20.png');


# ### Credit Value Adjustments

# In[99]:


S0 = 100.0
r = 0.05
sigma = 0.2
T = 1.0
I = 100000


# In[100]:


ST = S0 * np.exp(
    (r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * npr.standard_normal(I)
)


# In[101]:


L = 0.5


# In[102]:


p = 0.01


# In[103]:


D = npr.poisson(p * T, I)


# In[104]:


D = np.where(D > 1, 1, D)


# In[105]:


math.exp(-r * T) * np.mean(ST)


# In[106]:


CVaR = math.exp(-r * T) * np.mean(L * D * ST)
CVaR


# In[107]:


S0_CVA = math.exp(-r * T) * np.mean((1 - L * D) * ST)
S0_CVA


# In[108]:


S0_adj = S0 - CVaR
S0_adj


# In[109]:


np.count_nonzero(L * D * ST)


# In[110]:


plt.figure(figsize=(10, 6))
plt.hist(L * D * ST, bins=50)
plt.xlabel("loss")
plt.ylabel("frequency")
plt.ylim(ymax=175)
# plt.savefig('../../images/ch12/stoch_21.png');


# In[111]:


K = 100.0
hT = np.maximum(ST - K, 0)


# In[112]:


C0 = math.exp(-r * T) * np.mean(hT)
C0


# In[113]:


CVaR = math.exp(-r * T) * np.mean(L * D * hT)
CVaR


# In[114]:


C0_CVA = math.exp(-r * T) * np.mean((1 - L * D) * hT)
C0_CVA


# In[115]:


np.count_nonzero(L * D * hT)


# In[116]:


np.count_nonzero(D)


# In[117]:


I - np.count_nonzero(hT)


# In[118]:


plt.figure(figsize=(10, 6))
plt.hist(L * D * hT, bins=50)
plt.xlabel("loss")
plt.ylabel("frequency")
plt.ylim(ymax=350)
# plt.savefig('../../images/ch12/stoch_22.png');


# - https://home.tpq.io/

# In[ ]:
