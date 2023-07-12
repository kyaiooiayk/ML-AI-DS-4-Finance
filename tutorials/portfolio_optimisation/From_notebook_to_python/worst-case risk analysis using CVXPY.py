#!/usr/bin/env python
# coding: utf-8

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
#
# **What?** Worst-case analysis using CVXPY
#
# </font>
# </div>

# # Worst-case risk analysis

# ### Covariance uncertainty
#
# In this example we do worst-case risk analysis using CVXPY.
# Our setting is a single period Markowitz portfolio allocation problem.
# We have a fixed portfolio allocation $w \in {\bf R}^n$. The return covariance $\Sigma$ is not known,
# but we believe $\Sigma \in \mathcal S$.
# Here $\mathcal S$ is a convex set of possible covariance matrices.
# The risk is $w^T \Sigma w$, a linear function of $\Sigma$.
#
# We can compute the worst (maximum) risk, over all possible covariance matrices by solving the convex optimization
# problem
#
# $$
# \begin{array}{ll} \mbox{maximize} & w^T\Sigma w \\
# \mbox{subject to} & \Sigma \in \mathcal S, \quad \Sigma \succeq 0,
# \end{array}
# $$
#
# with variable $\Sigma$.
#
# If the worst-case risk is not too bad, you can worry less.
# If not, you'll confront your worst nightmare

# ### Example
#
# In the following code we solve the portfolio allocation problem
#
# $$
# \begin{array}{ll} \mbox{minimize} & w^T\Sigma_\mathrm{nom} w \\
# \mbox{subject to} & {\bf 1}^Tw = 1, \quad \mu^Tw \geq 0.1, \quad \|w\|_1 \leq 2,
# \end{array}
# $$
#
# and then compute the worst-case risk under the assumption that $\mathcal S = \left\{ \Sigma^\mathrm{nom} + \Delta \,:\,
# |\Delta_{ii}| =0, \;
# |\Delta_{ij}| \leq 0.2
# \right\}$.
#
# We might expect that $|\Delta_{ij}| = 0.2$ for all $i \neq j$.
# This does not happen however because of the constraint that $\Sigma^\mathrm{nom} + \Delta$ is positive semidefinite.

# In[ ]:


# Generate data for worst-case risk analysis.
import numpy as np

np.random.seed(2)
n = 5
mu = np.abs(np.random.randn(n, 1)) / 15
Sigma = np.random.uniform(-0.15, 0.8, size=(n, n))
Sigma_nom = Sigma.T.dot(Sigma)
print("Sigma_nom =")
print(np.round(Sigma_nom, decimals=2))


# In[ ]:


# Form and solve portfolio optimization problem.
# Here we minimize risk while requiring a 0.1 return.
import cvxpy as cp


w = cp.Variable(n)
ret = mu.T @ w
risk = cp.quad_form(w, Sigma_nom)
prob = cp.Problem(cp.Minimize(risk), [cp.sum(w) == 1, ret >= 0.1, cp.norm(w, 1) <= 2])
prob.solve()
print("w =")
print(np.round(w.value, decimals=2))


# In[ ]:


# Form and solve worst-case risk analysis problem.
Sigma = cp.Variable((n, n), PSD=True)
Delta = cp.Variable((n, n), symmetric=True)
risk = cp.quad_form(w.value, Sigma)
prob = cp.Problem(
    cp.Maximize(risk),
    [Sigma == Sigma_nom + Delta, cp.diag(Delta) == 0, cp.abs(Delta) <= 0.2],
)
prob.solve()
print("standard deviation =", cp.sqrt(cp.quad_form(w.value, Sigma_nom)).value)
print("worst-case standard deviation =", cp.sqrt(risk).value)
print("worst-case Delta =")
print(np.round(Delta.value, decimals=2))


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
#
# - https://www.cvxpy.org/examples/index.html#
#
# </font>
# </div>

# In[ ]:
