#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Credit-risk-and-bucketing" data-toc-modified-id="Credit-risk-and-bucketing-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Credit risk and bucketing</a></span></li><li><span><a href="#Load-dataset" data-toc-modified-id="Load-dataset-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Load dataset</a></span></li><li><span><a href="#EDA" data-toc-modified-id="EDA-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>EDA</a></span></li><li><span><a href="#Optimal-number-of-clusters" data-toc-modified-id="Optimal-number-of-clusters-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Optimal number of clusters</a></span><ul class="toc-item"><li><span><a href="#Elbow-method" data-toc-modified-id="Elbow-method-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Elbow method</a></span></li><li><span><a href="#Silhouette-method" data-toc-modified-id="Silhouette-method-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Silhouette method</a></span></li><li><span><a href="#Calinski-Harabasz-(CH)-method" data-toc-modified-id="Calinski-Harabasz-(CH)-method-6.3"><span class="toc-item-num">6.3&nbsp;&nbsp;</span>Calinski-Harabasz (CH) method</a></span></li><li><span><a href="#Gap-analysis" data-toc-modified-id="Gap-analysis-6.4"><span class="toc-item-num">6.4&nbsp;&nbsp;</span>Gap analysis</a></span></li></ul></li><li><span><a href="#Chosen-optimal-number-of-clusters" data-toc-modified-id="Chosen-optimal-number-of-clusters-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Chosen optimal number of clusters</a></span></li><li><span><a href="#Split-data" data-toc-modified-id="Split-data-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Split data</a></span><ul class="toc-item"><li><span><a href="#Logistic-Regression-for-PD-Estimation" data-toc-modified-id="Logistic-Regression-for-PD-Estimation-8.1"><span class="toc-item-num">8.1&nbsp;&nbsp;</span>Logistic Regression for PD Estimation</a></span></li><li><span><a href="#Bayesian-Approach-for-PD-Estimation" data-toc-modified-id="Bayesian-Approach-for-PD-Estimation-8.2"><span class="toc-item-num">8.2&nbsp;&nbsp;</span>Bayesian Approach for PD Estimation</a></span></li><li><span><a href="#Markov-Chain-for-PD-Estimation" data-toc-modified-id="Markov-Chain-for-PD-Estimation-8.3"><span class="toc-item-num">8.3&nbsp;&nbsp;</span>Markov Chain for PD Estimation</a></span></li><li><span><a href="#SVC-for-PD-Estimation" data-toc-modified-id="SVC-for-PD-Estimation-8.4"><span class="toc-item-num">8.4&nbsp;&nbsp;</span>SVC for PD Estimation</a></span></li><li><span><a href="#RF-for-PD-Estimation" data-toc-modified-id="RF-for-PD-Estimation-8.5"><span class="toc-item-num">8.5&nbsp;&nbsp;</span>RF for PD Estimation</a></span></li><li><span><a href="#NN-for-PD-Estimation" data-toc-modified-id="NN-for-PD-Estimation-8.6"><span class="toc-item-num">8.6&nbsp;&nbsp;</span>NN for PD Estimation</a></span></li><li><span><a href="#DL-for-PD-Estimation" data-toc-modified-id="DL-for-PD-Estimation-8.7"><span class="toc-item-num">8.7&nbsp;&nbsp;</span>DL for PD Estimation</a></span></li></ul></li><li><span><a href="#References" data-toc-modified-id="References-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Credit risk modelling
# 
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
from gap_statistic.optimalK import OptimalK
import seaborn as sns
import pymc3 as pm
import arviz as az
import logging
sns.set()
plt.rcParams["figure.figsize"] = (10,6)


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# # Credit risk and bucketing
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=bla ck>
# 
# - **Credit risk** is most simply defined as the potential that a bank borrower or counterparty will fail to meet its obligations in accordance with agreed terms. The goal of credit risk management is to maximise a bank’s risk-adjusted rate of return by maintaining credit risk exposure within acceptable parameters. The importance of having strong capital requirements for a bank rests on the idea that banks should have a capital buffer in turbulent times. Of course, ensuring at least a minimum capital requirement is a burden for financial institutions in the sense that capital is an asset they cannot channel to deficit entities to make a profit.                 
# The most important and challenging part of estimating credit risk is to model the probability of default.
# 
# 
# - **Risk bucketing** is nothing but grouping borrowers with similar creditworthiness. The behind-the-scenes story of risk bucketing is to obtain homogenous groups or clusters so that we can better estimate the credit risk. 
# Treating different risky borrowers equally may result in poor predictions because the model cannot capture entirely different characteristics of the data at once. 
# Thus, by dividing the borrowers into different groups based on riskiness, risk bucketing enables us to make accurate predictions.
# 
# </font>
# </div>

# # Load dataset
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=bla ck>
# 
# - Let’s create a practice exer‐ cise using German credit risk data. 
# - The data is gathered from the Kaggle platform.
# - The dataset includes both categorical and numerical values, which need to be treated differently.
# 
# </font>
# </div>

# In[2]:


credit = pd.read_csv('credit_data_risk.csv')


# # EDA
# <hr style = "border:2px solid black" ></hr>

# In[3]:


credit.head()


# In[4]:


del credit['Unnamed: 0']


# In[5]:


credit.describe()


# In[7]:


numerical_credit = credit.select_dtypes(exclude='O')


# In[8]:


plt.figure(figsize=(10, 8))
k = 0
cols = numerical_credit.columns
for i, j in zip(range(len(cols)), cols):
    k +=1
    plt.subplot(2, 2, k)
    plt.hist(numerical_credit.iloc[:, i])
    plt.title(j)


# # Optimal number of clusters
# <hr style = "border:2px solid black" ></hr>

# In[10]:


scaler = StandardScaler()
scaled_credit = scaler.fit_transform(numerical_credit)


# ## Elbow method

# <div class="alert alert-info">
# <font color=bla ck>
# 
# - the **elbow method** is based on the inertia.
# - The elbow method is used to find the optimal number of clusters, we observe the slope of the curve and decide the cut-off point at which the curve gets flatter—that is, the slope of the curve gets lower. 
# - As it gets flatter, the inertia, telling us how far away the points within a cluster are located, decreases, which is nice for the purpose of clustering. On the other hand, as we allow inertia to decrease, the number of clusters increases, which makes the analysis more complicated. Given that trade-off, the stopping criteria is the point where the **curve gets flatter**.
# 
# </font>
# </div>

# In[11]:


distance = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(scaled_credit)
    distance.append(kmeans.inertia_)


# In[12]:


plt.plot(range(1, 10), distance, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('The Elbow Method')
plt.show()


# ## Silhouette method

# <div class="alert alert-info">
# <font color=bla ck>
# 
# - The Silhouette score is introduced as a tool to decide the optimal number of clusters. This takes a value between 1 and -1. 
# - A value of 1 indicates that an observation is close to the correct centroid and correctly classified. However, -1 shows that an observation is not correctly clustered.
# - The strength of the Silhouette score rests on taking into account both the intracluster distance and the intercluster distance.
# 
# </font>
# </div>

# In[14]:


fig, ax = plt.subplots(4, 2, figsize=(25, 20))
for i in range(2, 10):
    km = KMeans(n_clusters=i)
    q, r = divmod(i, 2)
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick',
                                      ax=ax[q - 1][r])
    visualizer.fit(scaled_credit)
    ax[q - 1][r].set_title("For Cluster_"+str(i))
    ax[q - 1][r].set_xlabel("Silhouette Score")


# ## Calinski-Harabasz (CH) method

# <div class="alert alert-info">
# <font color=bla ck>
# 
# - Calinski-Harabasz (CH), which is known as the variance ratio criterion.
# - We are seeking a high CH score, as the larger (lower) the between-cluster variance (within cluster variance), the better it is for finding the optimal number of clusters.
# 
# </font>
# </div>

# In[15]:


from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2, 10),
                              metric='calinski_harabasz',
                              timings=False)
visualizer.fit(scaled_credit)
visualizer.show();


# ## Gap analysis

# <div class="alert alert-info">
# <font color=bla ck>
# 
# - The estimate of the optimal clusters will be the value that maximizes the **gap statistic**, as the gap statistic is the difference between the total within-intracluster variation for different values of k and their expected values under null reference distribution of the respective data. 
# - The decision is made when we get the highest gap value.
# 
# </font>
# </div>

# In[17]:


optimalK = OptimalK(n_jobs=8, parallel_backend='joblib')
n_clusters = optimalK(scaled_credit, cluster_array=np.arange(1, 10))


# In[18]:


gap_result = optimalK.gap_df
gap_result.head()


# In[19]:


plt.plot(gap_result.n_clusters, gap_result.gap_value)
min_ylim, max_ylim = plt.ylim()
plt.axhline(np.max(gap_result.gap_value), color='r',
            linestyle='dashed', linewidth=2)
plt.title('Gap Analysis')
plt.xlabel('Number of Cluster')
plt.ylabel('Gap Value')
plt.show()


# # Chosen optimal number of clusters
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=bla ck>
# 
# - In light of these discussions, two clusters are chosen to be the optimal number of clusters, and the K-means clustering analysis is conducted accordingly. 
# - To illustrate the result let's visualize 2-D clusters.
# 
# </font>
# </div>

# In[20]:


kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(scaled_credit)


# In[21]:


plt.figure(figsize=(10, 12))
plt.subplot(311)
plt.scatter(scaled_credit[:, 0], scaled_credit[:, 2],
            c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 2], s = 80,
            marker= 'x', color = 'k')
plt.title('Age vs Credit')
plt.subplot(312)
plt.scatter(scaled_credit[:, 0], scaled_credit[:, 2],
            c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 2], s = 80,
            marker= 'x', color = 'k')
plt.title('Credit vs Duration')
plt.subplot(313)
plt.scatter(scaled_credit[:, 2], scaled_credit[:, 3],
            c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 2],
            kmeans.cluster_centers_[:, 3], s = 120,
            marker= 'x', color = 'k')
plt.title('Age vs Duration')
plt.show()


# In[22]:


clusters, counts = np.unique(kmeans.labels_, return_counts=True)


# In[23]:


cluster_dict = {}
for i in range(len(clusters)):
    cluster_dict[i] = scaled_credit[np.where(kmeans.labels_==i)]


# In[24]:


credit['clusters'] = pd.DataFrame(kmeans.labels_)


# In[25]:


df_scaled = pd.DataFrame(scaled_credit)
df_scaled['clusters'] = credit['clusters']


# In[26]:


df_scaled['Risk'] = credit['Risk']
df_scaled.columns = ['Age', 'Job', 'Credit amount',
                     'Duration', 'Clusters', 'Risk']


# In[27]:


df_scaled[df_scaled.Clusters == 0]['Risk'].value_counts()


# In[28]:


df_scaled[df_scaled.Clusters == 1]['Risk'].value_counts()


# In[29]:


df_scaled[df_scaled.Clusters == 0]['Risk'].value_counts()                                    .plot(kind='bar',
                                    figsize=(10, 6),
                                    title="Frequency of Risk Level")
plt.show()


# In[30]:


df_scaled[df_scaled.Clusters == 1]['Risk'].value_counts()                                    .plot(kind='bar',
                                    figsize=(10, 6),
                                    title="Frequency of Risk Level")
plt.show()


# # Split data
# <hr style = "border:2px solid black" ></hr>

# In[83]:


from sklearn.model_selection import train_test_split


# In[84]:


df_scaled['Risk'] = df_scaled['Risk'].replace({'good': 1, 'bad': 0})


# In[85]:


X = df_scaled.drop('Risk', axis=1)
y = df_scaled.loc[:, ['Risk', 'Clusters']]


# In[86]:


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)


# In[87]:


first_cluster_train = X_train[X_train.Clusters == 0].iloc[:, :-1]
second_cluster_train = X_train[X_train.Clusters == 1].iloc[:, :-1]


# ## Logistic Regression for PD Estimation

# In[36]:


import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')


# In[37]:


X_train1 = first_cluster_train
y_train1 = y_train[y_train.Clusters == 0]['Risk']
smote = SMOTEENN(random_state = 2)
X_train1, y_train1 = smote.fit_resample(X_train1, y_train1.ravel())
logit = sm.Logit(y_train1, X_train1)
logit_fit1 = logit.fit()
print(logit_fit1.summary())


# In[38]:


first_cluster_test = X_test[X_test.Clusters == 0].iloc[:, :-1]
second_cluster_test = X_test[X_test.Clusters == 1].iloc[:, :-1]


# In[39]:


X_test1 = first_cluster_test
y_test1 = y_test[y_test.Clusters == 0]['Risk']
pred_prob1 = logit_fit1.predict(X_test1)


# In[40]:


false_pos, true_pos, _ = roc_curve(y_test1.values,  pred_prob1)
auc = roc_auc_score(y_test1, pred_prob1)
plt.plot(false_pos,true_pos, label="AUC for cluster 1={:.4f} "
         .format(auc))
plt.plot([0, 1], [0, 1], linestyle = '--', label='45 degree line')
plt.legend(loc='best')
plt.title('AUC-ROC Curve 1')
plt.show()


# In[41]:


X_train2 = second_cluster_train
y_train2 = y_train[y_train.Clusters == 1]['Risk']
logit = sm.Logit(y_train2, X_train2)
logit_fit2 = logit.fit()
print(logit_fit2.summary())


# In[42]:


X_test2 = second_cluster_test
y_test2 = y_test[y_test.Clusters == 1]['Risk']
pred_prob2 = logit_fit2.predict(X_test2)


# In[43]:


false_pos, true_pos, _ = roc_curve(y_test2.values,  pred_prob2)
auc = roc_auc_score(y_test2, pred_prob2)
plt.plot(false_pos,true_pos,label="AUC for cluster 2={:.4f} "
         .format(auc))
plt.plot([0, 1], [0, 1], linestyle = '--', label='45 degree line')
plt.legend(loc='best')
plt.title('AUC-ROC Curve 2')
plt.show()


# ## Bayesian Approach for PD Estimation

# <div class="alert alert-info">
# <font color=bla ck>
# 
# - `PYMC3` package, which is a Python package for Bayesian estimation, to predict the probability of default. 
# - There are several approaches for running Bayesian analysis using `PYMC3`, and for the first application, we’ll use the MAP distribution discussed. As a quick reminder, given the representative posterior distribution, MAP becomes an efficient model in this case. 
# - Moreover, we select the Bayesian model with a deterministic variable (p) that is entirely determined by its parents—that is, age, job, credit amount, and duration.
# 
# </font>
# </div>

# In[47]:


# Identifying Bayesian model as logistic_model1
with pm.Model() as logistic_model1:
    
    #dentifying the assumed distributions of the variables as 
    #normal with defined mu and sigma parameters
    beta_age = pm.Normal('coeff_age', mu=0, sd=10)
    beta_job = pm.Normal('coeff_job', mu=0, sd=10)
    beta_credit = pm.Normal('coeff_credit_amount', mu=0, sd=10)
    beta_dur = pm.Normal('coeff_duration', mu=0, sd=10)
    
    # Running a deterministic model using the first sample
    p = pm.Deterministic('p', pm.math.sigmoid(beta_age * 
                              X_train1['Age'] + beta_job *
                              X_train1['Job'] + beta_credit *
                              X_train1['Credit amount'] + beta_dur *
                              X_train1['Duration']))
    

with logistic_model1:
    # Running a Bernoulli distribution to model the dependent variable
    observed = pm.Bernoulli("risk", p, observed=y_train1)
    # Fitting the MAP model to data
    map_estimate = pm.find_MAP()


# In[48]:


param_list = ['coeff_age', 'coeff_job',
              'coeff_credit_amount', 'coeff_duration']
params = {}
for i in param_list:
    params[i] = [np.round(map_estimate[i], 6)] 
    
bayesian_params = pd.DataFrame.from_dict(params)    
print('The result of Bayesian estimation:\n {}'.format(bayesian_params))


# In[49]:


with pm.Model() as logistic_model2:
    beta_age = pm.Normal('coeff_age', mu=0, sd=10)
    beta_job = pm.Normal('coeff_job', mu=0, sd=10)
    beta_credit = pm.Normal('coeff_credit_amount', mu=0, sd=10)
    beta_dur = pm.Normal('coeff_duration', mu=0, sd=10)
    p = pm.Deterministic('p', pm.math.sigmoid(beta_age *
                              second_cluster_train['Age'] + 
                              beta_job * second_cluster_train['Job'] + 
                              beta_credit * second_cluster_train['Credit amount'] + 
                              beta_dur * second_cluster_train['Duration']))
with logistic_model2:
    observed = pm.Bernoulli("risk", p,
                            observed=y_train[y_train.Clusters == 1]
                            ['Risk'])
    map_estimate = pm.find_MAP()


# In[50]:


param_list = [ 'coeff_age', 'coeff_job',
              'coeff_credit_amount', 'coeff_duration']
params = {}
for i in param_list:
    params[i] = [np.round(map_estimate[i], 6)]
    
bayesian_params = pd.DataFrame.from_dict(params)    
print('The result of Bayesian estimation:\n {}'.format(bayesian_params))


# ## Markov Chain for PD Estimation

# <div class="alert alert-info">
# <font color=bla ck>
# 
# - Instead of finding the local maximum, which is sometimes difficult to get, we look for an approximate expectation based on the sampling procedure. This is referred to as MCMC in the Bayesian setting. One of the most well known methods is the Metropolis-Hastings (M-H) algorithm.
# - Accordingly, we draw 10,000 posterior samples to simulate the posterior distribution for two independent Markov chains.
# 
# </font>
# </div>

# In[51]:


# Importing the logging package to suppress the warning messages
logger = logging.getLogger('pymc3')
logger.setLevel(logging.ERROR)


# In[52]:


with logistic_model1:
    step = pm.Metropolis()
    trace = pm.sample(10000, step=step,progressbar = False)
az.plot_trace(trace)
plt.show()


# In[53]:


with logistic_model1:
    display(az.summary(trace, round_to=6)[:4])


# In[54]:


with logistic_model2:
    step = pm.Metropolis()
    trace = pm.sample(10000, step=step,progressbar = False)
az.plot_trace(trace)
plt.show()


# In[55]:


with logistic_model2:
    display(az.summary(trace, round_to=6)[:4])


# <div class="alert alert-info">
# <font color=bla ck>
# 
# - One disadvantage of the M-H algorithm is its sensitivity to step size. 
# - Small steps hinder the convergence process. Conversely, big steps may cause a high rejection rate. 
# - Besides, M-H may suffer from rare events—as the probability of these events are low, requiring a large sample to obtain a reliable estimation—and that is our focus in this case.
# 
# </font>
# </div>

# ## SVC for PD Estimation

# In[56]:


from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
import time


# In[57]:


param_svc = {'gamma': [1e-6, 1e-2],
             'C':[0.001,.09,1,5,10],
             'kernel':('linear','rbf')}


# In[58]:


svc = SVC(class_weight='balanced')
halve_SVC = HalvingRandomSearchCV(svc, param_svc, 
                                  scoring = 'roc_auc', n_jobs=-1)
halve_SVC.fit(X_train1, y_train1)
print('Best hyperparameters for first cluster in SVC {} with {}'.
      format(halve_SVC.best_score_, halve_SVC.best_params_))


# In[59]:


y_pred_SVC1 = halve_SVC.predict(X_test1)
print('The ROC AUC score of SVC for first cluster is {:.4f}'.
      format(roc_auc_score(y_test1, y_pred_SVC1)))


# In[60]:


halve_SVC.fit(X_train2, y_train2)
print('Best hyperparameters for second cluster in SVC {} with {}'.
      format(halve_SVC.best_score_, halve_SVC.best_params_))


# In[61]:


y_pred_SVC2 = halve_SVC.predict(X_test2)
print('The ROC AUC score of SVC for first cluster is {:.4f}'.
      format(roc_auc_score(y_test2, y_pred_SVC2)))


# ## RF for PD Estimation

# In[62]:


from sklearn.ensemble import RandomForestClassifier


# In[63]:


rfc = RandomForestClassifier(random_state=42)


# In[64]:


param_rfc = {'n_estimators': [100, 300],
    'criterion' :['gini', 'entropy'],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [3, 4, 5, 6],
    'min_samples_split':[5, 10]}


# In[65]:


halve_RF = HalvingRandomSearchCV(rfc, param_rfc,
                                 scoring = 'roc_auc', n_jobs=-1)
halve_RF.fit(X_train1, y_train1)
print('Best hyperparameters for first cluster in RF {} with {}'.
      format(halve_RF.best_score_, halve_RF.best_params_))


# In[66]:


y_pred_RF1 = halve_RF.predict(X_test1)
print('The ROC AUC score of RF for first cluster is {:.4f}'.
      format(roc_auc_score(y_test1, y_pred_RF1)))


# In[67]:


halve_RF.fit(X_train2, y_train2)
print('Best hyperparameters for second cluster in RF {} with {}'.
      format(halve_RF.best_score_, halve_RF.best_params_))


# In[68]:


y_pred_RF2 = halve_RF.predict(X_test2)
print('The ROC AUC score of RF for first cluster is {:.4f}'.
      format(roc_auc_score(y_test2, y_pred_RF2)))


# ## NN for PD Estimation

# In[69]:


from sklearn.neural_network import MLPClassifier


# In[70]:


param_NN = {"hidden_layer_sizes": [(100, 50), (50, 50), (10, 100)],
            "solver": ["lbfgs", "sgd", "adam"], 
            "learning_rate_init": [0.001, 0.05]}


# In[71]:


MLP = MLPClassifier(random_state=42)


# In[72]:


param_halve_NN = HalvingRandomSearchCV(MLP, param_NN,
                                       scoring = 'roc_auc')
param_halve_NN.fit(X_train1, y_train1)
print('Best hyperparameters for first cluster in NN are {}'.
      format(param_halve_NN.best_params_))


# In[73]:


y_pred_NN1 = param_halve_NN.predict(X_test1)
print('The ROC AUC score of NN for first cluster is {:.4f}'.
      format(roc_auc_score(y_test1, y_pred_NN1)))


# In[74]:


param_halve_NN.fit(X_train2, y_train2)
print('Best hyperparameters for first cluster in NN are {}'.
      format(param_halve_NN.best_params_))


# In[75]:


y_pred_NN2 = param_halve_NN.predict(X_test2)
print('The ROC AUC score of NN for first cluster is {:.4f}'.
      format(roc_auc_score(y_test2, y_pred_NN2)))


# ## DL for PD Estimation

# In[76]:


from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)


# In[77]:


def DL_risk(dropout_rate,verbose=0):
    model = keras.Sequential()
    model.add(Dense(128,kernel_initializer='normal', 
        activation = 'relu', input_dim=4))
    model.add(Dense(64, kernel_initializer='normal', 
        activation = 'relu'))
    model.add(Dense(8,kernel_initializer='normal', 
        activation = 'relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return model


# In[78]:


parameters = {'batch_size':  [10, 50, 100],
          'epochs':  [50, 100, 150],
             'dropout_rate':[0.2, 0.4]}
model = KerasClassifier(build_fn = DL_risk)
gs = GridSearchCV(estimator = model,
                       param_grid = parameters,
                          scoring = 'roc_auc')


# In[79]:


gs.fit(X_train1, y_train1, verbose=0)
print('Best hyperparameters for first cluster in DL are {}'.
      format(gs.best_params_))


# In[80]:


model = KerasClassifier(build_fn = DL_risk,
                        dropout_rate = gs.best_params_['dropout_rate'],
                        verbose = 0,
                        batch_size = gs.best_params_['batch_size'],
                        epochs = gs.best_params_['epochs'])
model.fit(X_train1, y_train1)
DL_predict1 = model.predict(X_test1)
DL_ROC_AUC = roc_auc_score(y_test1, pd.DataFrame(DL_predict1.flatten()))
print('DL_ROC_AUC is {:.4f}'.format(DL_ROC_AUC))


# In[81]:


gs.fit(X_train2.values, y_train2.values, verbose=0)
print('Best parameters for second cluster in DL are {}'.
      format(gs.best_params_))


# In[82]:


model = KerasClassifier(build_fn = DL_risk,
                        dropout_rate= gs.best_params_['dropout_rate'],
                        verbose = 0,
                        batch_size = gs.best_params_['batch_size'],
                        epochs = gs.best_params_['epochs'])
model.fit(X_train2, y_train2)
DL_predict2 =  model.predict(X_test2)
DL_ROC_AUC = roc_auc_score(y_test2, DL_predict2.flatten()) 
print('DL_ROC_AUC is {:.4f}'.format(DL_ROC_AUC))


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - Machine Learning for Financial Risk Management with Python Abdullah Karasan
# - [Kaggle dataset download link](https://www.kaggle.com/datasets/uciml/german-credit)
# - https://github.com/abdullahkarasan/mlfrm/blob/main/codes/chp_6.ipynb
#     
# </font>
# </div>

# In[ ]:




