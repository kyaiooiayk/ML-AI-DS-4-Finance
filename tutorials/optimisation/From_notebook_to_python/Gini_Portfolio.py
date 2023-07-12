#!/usr/bin/env python
# coding: utf-8

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# # Gini Mean Difference Portfolio Optimization
#
# In this notebook we show how we can solve a hard problem using some reformulations.
#
# ## 1. Gini Optimization
#
# ### 1.1 Original formulation
#
# The Gini mean difference (GMD) is a measure of dispersion and it was introduced in the context of portfolio optimization by __[Yitzhaki (1982)](https://www.researchgate.net/publication/4900733_Stochastic_Dominance_Mean_Variance_and_Gini%27s_Mean_Difference)__. However, this model is not used by practitioners due to the original formulation having a number of variables that increases proportional to $T(T-1)/2$, where $T$ is the number of observations. The original model is presented as follows:
#
# $$
# \begin{equation}
# \begin{aligned}
# & \underset{x,\, d}{\text{min}} & & \frac{1}{T(T-1)} \sum^{T}_{i=1} \sum^{T}_{j > i} d_{i,j} \\
# & \text{s.t.} & & \mu x \geq \bar{\mu} \\
# & & &  d_{i,j} \geq (r_{i} -r_{j})x \; ; \; \forall \; i,j =1, \ldots, T \; ; \; i < j \\
# & & &  d_{i,j} \geq -(r_{i} -r_{j})x \\
# & & &  \sum_{i=1}^{N} x_i = 1 \\
# & & &  x_{i} \geq 0 \; ; \; \forall \; i =1, \ldots, N \\
# & & &  d_{i,j} \geq 0 \; ; \; \forall \; i,j =1, \ldots, T \\
# \end{aligned}
# \end{equation}
# $$
#
# Where $r_{i}$ is the vector of returns in period $i$ and $d$ is an auxiliary variable.
#
# ### 1.2 Murray's reformulation
#
# To increase the efficiency of the problem above, __[Murray (2022)](https://github.com/cvxpy/cvxpy/issues/1585)__ proposed the following reformulation:
#
# $$
# \begin{equation}
# \begin{aligned}
# & \underset{x,\, d}{\text{min}} & & \frac{1}{T(T-1)} \sum^{T}_{i=1} \sum^{T}_{j > i} d_{i,j} \\
# & \text{s.t.} & & \mu x \geq \bar{\mu} \\
# & & & y = r x \\
# & & & d \geq M y \\
# & & & d \geq -M y \\
# & & &  \sum_{i=1}^{N} x_i = 1 \\
# & & &  x_{i} \geq 0 \; ; \; \forall \; i =1, \ldots, N \\
# & & &  d_{i,j} \geq 0 \; ; \; \forall \; i,j =1, \ldots, T \\
# \end{aligned}
# \end{equation}
# $$
#
# where
# $$
# M = \begin{bmatrix}
# \left. \begin{matrix}
# -1 & 1 & 0 & 0 & \ldots & 0 & 0\\
# -1 & 0 & 1 & 0 & \ldots & 0 & 0\\
# \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
# -1 & 0 & 0 & 0 & \ldots & 0 & 1 \\
# \end{matrix} \right \} & T-1\\
# \left. \begin{matrix}
# 0 & -1 & 1 & 0 & \ldots & 0 & 0\\
# 0 &  -1 & 0 & 1 & \ldots & 0 & 0\\
# \vdots & \vdots & \vdots & \vdots & \ddots & \vdots &  \vdots \\
# 0 & -1 & 0 & 0 & \ldots & 0 & 1 \\
# \end{matrix} \right \} & T-2\\
# \vdots \\
# \underbrace{ \left. \begin{matrix}
# 0 & 0 & 0 & 0 & \ldots & -1 & 1 \\
# \end{matrix} \right \} }_{T} & 1 \\
# \end{bmatrix}
# $$
#
# This reformulation is more efficient for medium scale problems (T<800).
#
# ### 1.3 Cajas's reformulation:
#
# __[Cajas (2021)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3988927)__ proposed an alternative reformulation based on the ordered weighted averaging (OWA) operator optimization model for monotonic weights proposed by __[Chassein and Goerigk (2015)](https://kluedo.ub.uni-kl.de/frontdoor/deliver/index/docId/3899/file/paper.pdf)__. This formulation works better for large scale problems (T>=800). This formulation is presented as follows:
#
# $$
# \begin{equation}
# \begin{aligned}
# & \min_{\alpha, \, \beta, \, x, \, y} & & \sum^{T}_{i=1} \alpha_{i} + \beta_{i}  \\
# & \text{s.t.} & & \mu x \geq \bar{\mu} \\
# & & & r x = y \\
# & & & \alpha_{i} + \beta_{j} \geq w_{i} y_{j}  \;\;\;\; \forall \; i,j =1, \ldots, T \\
# & & &  \sum_{i=1}^{N} x_i = 1 \\
# & & &  x_i \geq 0 \; ; \; \forall \; i =1, \ldots, N \\
# \end{aligned}
# \end{equation}
# $$
#
# where $w_{i} =  2 \left ( \frac{2i - 1 - T}{T(T-1)} \right )$.

# In[ ]:


import numpy as np
import pandas as pd
import cvxpy as cp
import mosek
import scipy.stats as st
from timeit import default_timer as timer
from datetime import timedelta
import warnings

warnings.filterwarnings("ignore")


def gini(mu, returns, D, assets, lift=0):
    (T, N) = returns.shape

    d = cp.Variable((int(T * (T - 1) / 2), 1))
    w = cp.Variable((N, 1))
    constraints = []

    if lift in ["Murray", "Yitzhaki"]:  # use Murray's reformulation
        if lift == "Murray":
            ret_w = cp.Variable((T, 1))
            constraints.append(ret_w == returns @ w)
            mat = np.zeros((d.shape[0], T))
            """ 
            We need to create a vector that has the following entries:
                ret_w[i] - ret_w[j]
            for j in range(T), for i in range(j+1, T).
            We do this by building a numpy array of mostly 0's and 1's.
            (It would be better to use SciPy sparse matrix objects.)
            """
            ell = 0
            for j in range(T):
                for i in range(j + 1, T):
                    # write to mat so that (mat @ ret_w)[ell] == var_i - var_j
                    mat[ell, i] = 1
                    mat[ell, j] = -1
                    ell += 1
            all_pairs_ret_diff = mat @ ret_w
        elif lift == "Yitzhaki":  # use the original formulation
            all_pairs_ret_diff = D @ w

        constraints += [
            d >= all_pairs_ret_diff,
            d >= -all_pairs_ret_diff,
            w >= 0,
            cp.sum(w) == 1,
        ]

        risk = cp.sum(d) / ((T - 1) * T)

    elif lift == "Cajas":
        a = cp.Variable((T, 1))
        b = cp.Variable((T, 1))
        y = cp.Variable((T, 1))

        owa_w = []
        for i in range(1, T + 1):
            owa_w.append(2 * i - 1 - T)
        owa_w = np.array(owa_w) / (T * (T - 1))

        constraints = [returns @ w == y, w >= 0, cp.sum(w) == 1]

        for i in range(T):
            constraints += [a[i] + b >= cp.multiply(owa_w[i], y)]

        risk = cp.sum(a + b)

    objective = cp.Minimize(risk * 1000)

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(
            solver=cp.MOSEK,
            mosek_params={mosek.iparam.num_threads: 2},
            verbose=False,
        )
        w = pd.DataFrame(w.value)
        w.index = assets
        w = w / w.sum()
    except:
        w = None
    return w


# In[ ]:


####################################
# Calculating the Gini Portfolio
# using three formulations
####################################

rs = np.random.RandomState(123)

sizes = [100]
data = {}
weights = {}
lifts = ["Yitzhaki", "Murray", "Cajas"]
k = 0
T = [200, 500, 700, 1000]

for t in T:
    for n in sizes:
        cov = rs.rand(n, n) * 1.5 - 0.5
        cov = cov @ cov.T / 1000 + np.diag(rs.rand(n) * 0.7 + 0.3) / 1000
        mean = np.zeros(n) + 1 / 1000

        Y = st.multivariate_normal.rvs(mean=mean, cov=cov, size=t, random_state=rs)
        Y = pd.DataFrame(Y)
        assets = ["Asset " + str(i) for i in range(1, n + 1)]
        mu = Y.mean().to_numpy()
        returns = Y.to_numpy()

        D = np.array([]).reshape(0, len(assets))
        for j in range(0, returns.shape[0] - 1):
            D = np.concatenate((D, returns[j + 1 :] - returns[j, :]), axis=0)

        for lift in lifts:
            name = str(lift) + "-" + str(t) + "-" + str(n)
            data[name] = []
            weights[name] = []
            if t >= 700 and lift == "Yitzhaki":
                continue
            else:
                start = timer()
                w = gini(mu, returns, D, assets, lift=lift)
                end = timer()
                data[name].append(timedelta(seconds=end - start).total_seconds())
                weights[name].append(w)

            k += 1
            print(name)


# In[ ]:


keys = list(data.keys())
for i in keys:
    if len(data[i]) == 0:
        del data[i]

pd.options.display.float_format = "{:.2f}".format

a = pd.DataFrame(data).T
a.columns = ["Time in Seconds"]
display(a)


# As we can see, as the number of observations $T$ increases the formulation proposed by Cajas (2021) becomes more efficient than Yitzhaki's and Murray's formulations.

# In[ ]:


b = pd.DataFrame([])
for i in weights.keys():
    if len(weights[i]) == 0:
        continue
    weights[i][0].columns = [i]
    b = pd.concat([b, mu @ weights[i][0]], axis=0)

pd.options.display.float_format = "{:.4%}".format

display(b)


# If we calculate the expected returns for each formulation, we can see that the three models give us the same results, which means that these formulations give us the same solution.

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
