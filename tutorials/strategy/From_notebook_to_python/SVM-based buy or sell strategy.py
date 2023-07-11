#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Load-data" data-toc-modified-id="Load-data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Load data</a></span></li><li><span><a href="#EDA" data-toc-modified-id="EDA-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>EDA</a></span></li><li><span><a href="#Features-engineering---indicators" data-toc-modified-id="Features-engineering---indicators-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Features engineering - indicators</a></span></li><li><span><a href="#Clean/trim-the-data" data-toc-modified-id="Clean/trim-the-data-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Clean/trim the data</a></span></li><li><span><a href="#Train-and-test-data" data-toc-modified-id="Train-and-test-data-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Train and test data</a></span></li><li><span><a href="#Create-output-signals" data-toc-modified-id="Create-output-signals-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Create output signals</a></span></li><li><span><a href="#Create-features-and-target" data-toc-modified-id="Create-features-and-target-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Create features and target</a></span><ul class="toc-item"><li><span><a href="#Find-best-parameters" data-toc-modified-id="Find-best-parameters-9.1"><span class="toc-item-num">9.1&nbsp;&nbsp;</span>Find best parameters</a></span><ul class="toc-item"><li><span><a href="#Pipeline-and-functions" data-toc-modified-id="Pipeline-and-functions-9.1.1"><span class="toc-item-num">9.1.1&nbsp;&nbsp;</span>Pipeline and functions</a></span></li><li><span><a href="#Hyperparameters" data-toc-modified-id="Hyperparameters-9.1.2"><span class="toc-item-num">9.1.2&nbsp;&nbsp;</span>Hyperparameters</a></span></li><li><span><a href="#Training-on-and-fetching-the-best-parameters" data-toc-modified-id="Training-on-and-fetching-the-best-parameters-9.1.3"><span class="toc-item-num">9.1.3&nbsp;&nbsp;</span>Training on and fetching the best parameters</a></span></li></ul></li><li><span><a href="#Create-the-Support-Vector-Machine" data-toc-modified-id="Create-the-Support-Vector-Machine-9.2"><span class="toc-item-num">9.2&nbsp;&nbsp;</span>Create the Support Vector Machine</a></span><ul class="toc-item"><li><span><a href="#Train-the-data" data-toc-modified-id="Train-the-data-9.2.1"><span class="toc-item-num">9.2.1&nbsp;&nbsp;</span>Train the data</a></span></li></ul></li><li><span><a href="#Predict-the-signals" data-toc-modified-id="Predict-the-signals-9.3"><span class="toc-item-num">9.3&nbsp;&nbsp;</span>Predict the signals</a></span><ul class="toc-item"><li><span><a href="#Save-the-predictions" data-toc-modified-id="Save-the-predictions-9.3.1"><span class="toc-item-num">9.3.1&nbsp;&nbsp;</span>Save the predictions</a></span></li></ul></li><li><span><a href="#Use-the-model-for-trading-strategy" data-toc-modified-id="Use-the-model-for-trading-strategy-9.4"><span class="toc-item-num">9.4&nbsp;&nbsp;</span>Use the model for trading strategy</a></span><ul class="toc-item"><li><span><a href="#Trading-strategy" data-toc-modified-id="Trading-strategy-9.4.1"><span class="toc-item-num">9.4.1&nbsp;&nbsp;</span>Trading strategy</a></span></li></ul></li><li><span><a href="#Analyze-the-performance" data-toc-modified-id="Analyze-the-performance-9.5"><span class="toc-item-num">9.5&nbsp;&nbsp;</span>Analyze the performance</a></span></li><li><span><a href="#Plot-the-results" data-toc-modified-id="Plot-the-results-9.6"><span class="toc-item-num">9.6&nbsp;&nbsp;</span>Plot the results</a></span></li></ul></li><li><span><a href="#References" data-toc-modified-id="References-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style="border:2px solid black"> </hr>

# - **What?** SVM buy or sell strategy

# # Imports
# <hr style="border:2px solid black"> </hr>

# In[1]:


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#!pip install TA-lib
import talib as ta


# # Load data
# <hr style="border:2px solid black"> </hr>

# In[2]:


Df = pd.read_csv("../datasource/random_stock_data.csv")
Df.head(30)


# # EDA
# <hr style="border:2px solid black"> </hr>

# In[3]:


# Drop the rows with 0 volume traded
Df = Df.drop(Df[Df["Volume"] == 0].index)


# In[4]:


# Convert the 'Time' column into pandas datetime format
Df["Time"] = pd.to_datetime(Df["Time"])


# In[5]:


Df["Close"].pct_change().plot(kind="hist", bins=100, figsize=(10, 5))


# # Features engineering - indicators
# <hr style="border:2px solid black"> </hr>

# - **Relative Strength Index (RSI)** The RSI provides technical traders with signals about bullish and bearish price momentum, and it is often plotted beneath the graph of an asset’s price.
# - **Simple Moving Average (SMA)** s an arithmetic moving average calculated by adding recent prices and then dividing that figure by the number of time periods in the calculation average.
# - **The parabolic SAR (stop and reversal)** is a technical indicator used to determine the price direction of an asset, as well as draw attention to when the price direction is changing. Sometimes known as the "stop and reversal system".
# - **Average Directional Index (ADX)**  makes use of a positive (+DI) and negative (-DI) directional indicator in addition to the trendline. The trend has strength when ADX is above 25; the trend is weak or the price is trendless when ADX is below 20,
#

# In[6]:


# Create a variable n with a value of 10 = 10 Minutes
n = 10


# In[7]:


# Create a column by name, RSI and assign the calculation of RSI to it
Df["RSI"] = ta.RSI(np.array(Df["Close"].shift(1)), timeperiod=n)


# In[8]:


# Create a column by name, SMA and assign the SMA calculation to it
Df["SMA"] = Df["Close"].shift(1).rolling(window=n).mean()

# Create a column by name, Corr and assign the calculation of correlation to it
Df["Corr"] = Df["Close"].shift(1).rolling(window=n).corr(Df["SMA"].shift(1))


# In[9]:


# Create a column by name, SAR and assign the SAR calculation to it
Df["SAR"] = ta.SAR(
    np.array(Df["High"].shift(1)), np.array(Df["Low"].shift(1)), 0.2, 0.2
)

# Create a column by name, ADX and assign the ADX calculation to it
Df["ADX"] = ta.ADX(
    np.array(Df["High"].shift(1)),
    np.array(Df["Low"].shift(1)),
    np.array(Df["Open"]),
    timeperiod=n,
)


# We will pass yesterday's "High", "Low", and "Open" prices as input to the algorithm in variables named in lower cases. This will help the algorithm sense the volatility of the past time period.

# In[10]:


# Create columns 'high', 'low' and 'close' with previous day's OHLC data
Df["high"] = Df["High"].shift(1)
Df["low"] = Df["Low"].shift(1)
Df["close"] = Df["Close"].shift(1)


# We will also create two more columns as features: the change in "Open" prices between yesterday and today & the difference between today's "Open" and yesterday's "Close" prices.

# In[11]:


# Create columns 'OO' with the difference between today's open and previous day's open
Df["OO"] = Df["Open"] - Df["Open"].shift(1)

# Create columns 'OC' with the difference between today's open and previous day's close
Df["OC"] = Df["Open"] - Df["close"]


# In[12]:


# Create a column 'Ret' with calculation of returns
Df["Ret"] = (Df["Open"].shift(-1) - Df["Open"]) / Df["Open"]

# Create n columns and assign
for i in range(1, n):
    Df["return%i" % i] = Df["Ret"].shift(i)


# # Clean/trim the data
# <hr style="border:2px solid black"> </hr>

# You need to keep the values of indicator 'Corr' between -1 and 1, as the correlation coefficient is always between these values. This is done by changing all values less than -1 to -1, and all values greater than 1 to 1.
#
# This doesn't affect our calculations negatively because the extreme values are realised due to NAN values in the data, which need to be handled before training the algorithm. Then we drop all NANs from the entire dataframe.

# In[13]:


# Change the value of 'Corr' to -1 if it is less than -1
Df.loc[Df["Corr"] < -1, "Corr"] = -1

# Change the value of 'Corr' to 1 if it is greater than 1
Df.loc[Df["Corr"] > 1, "Corr"] = 1

# Drop the NAN values
Df = Df.dropna()


# In[14]:


Df


# # Train and test data
# <hr style="border:2px solid black"> </hr>

# We will be using 80% of the data to train and the rest 20% to test. To do this, you will create a split parameter which will divide the dataframe in an 80-20 ratio.
#
# This can be changed as per your choice, but it is advisable to give at least 70% data as train data for good results. "split" is the integer index value for the row corresponding to test-train split.

# In[15]:


# Create a variable split which is 80% of the length of the Dataframe
t = 0.8
split = int(t * len(Df))
split


# # Create output signals
# <hr style="border:2px solid black"> </hr>

# Next, assign signal values corresponding to 'returns' that were calculated earlier. To do this, you will split the data into three equal parts, using the split on 'Ret' column.
# 1. Highest returns’ quantile is assigned Signal '1' or "Buy".
# 2. Middle quantile is assigned Signal '0' or 'Do nothing'.
# 3. Lowest quantile is assigned Signal '-1' or 'Sell'.

# In[16]:


# Create a column by name, 'Signal' and initialize with 0
Df["Signal"] = 0

# Assign a value of 1 to 'Signal' column for the quantile with highest returns
Df.loc[Df["Ret"] > Df["Ret"][:split].quantile(q=0.66), "Signal"] = 1

# Assign a value of -1 to 'Signal' column for the quantile with lowest returns
Df.loc[Df["Ret"] < Df["Ret"][:split].quantile(q=0.34), "Signal"] = -1


# In[17]:


# Assign a value of 0 to 'Signal' column at 1529 time
Df.loc[(Df["Time"].dt.hour == 15) & (Df["Time"].dt.minute == 29), "Signal"] = 0

# Assign a value of 0 to 'Ret' column at 1529 time
Df.loc[(Df["Time"].dt.hour == 15) & (Df["Time"].dt.minute == 29), "Ret"] = 0


# # Create features and target
# <hr style="border:2px solid black"> </hr>

# Drop the columns 'Close', 'Signal', 'Time', 'High', 'Low', 'Volume', and 'Ret' since the algorithm will not be trained on these features. Next, we assign 'Signal' to 'y' which is the output variable that you will predict using test data.

# In[18]:


# Use df.drop() to drop the columns
X = Df.drop(["Close", "Signal", "Time", "High", "Low", "Volume", "Ret"], axis=1)

# Create a variable which contains all the 'Signal' values
y = Df["Signal"]


# In[19]:


# Plot them together
pd.concat([X, y], axis=1)


# ## Find best parameters

# ### Pipeline and functions
#
# As the very first step to finding the best hyperparameters among C, Gamma and Kernel, you will first create a pipeline of functions which are required to run in a certain order on the training data.

# The ‘steps’ contains references to functions that would be applied to the data when called through a pipeline function. In this case, you will scale the data first and then fit it to the SVC function. This is done to avoid the effect of the individual weights of the features.

# In[20]:


# Create the 'steps' variable with the pipeline functions
steps = [("scaler", StandardScaler()), ("svc", SVC())]

# Pass the 'steps' to the Pipeline function
pipeline = Pipeline(steps)


# ### Hyperparameters
#
# The hyperparameters are iterated over to arrive at the best possible combination for the given training data. These test values can be changed as per your choice. Here, you will choose 4 test values for 'c' and 3 test values for 'g'.

# In[21]:


# Test variables for 'c' and 'g'
c = [10, 100, 1000, 10000]
g = [1e-2, 1e-1, 1e0]


# The 'rbf' is used as a singular entry in the kernel parameters. But you can go ahead and try other kernel functions, such as linear, poly and sigmoid.
#
# <b>Do remember:</b> A higher number of parameters would result in a greater time for the code to run.

# In[22]:


# Intialize the parameters
parameters = {"svc__C": c, "svc__gamma": g, "svc__kernel": ["rbf", "poly", "sigmoid"]}


# Next, you need to create a RandomizedSearchCV function with a cross validation value of 7. This value can be anything more than or equal to 3. The concept of cross validation is used to arrive at the scores of different random combinations of the hyperparameters.
#
# These scores would be used to find the best parameters and create a newly optimized support vector classifier.

# In[23]:


# Call the RandomizedSearchCV function and pass the parameters
rcv = RandomizedSearchCV(pipeline, parameters, cv=7)  # , iid=False)


# ### Training on and fetching the best parameters
#
# Next, you need to fit the train data to 'rcv' created above to obtain the best hyperparameters. The best parameters can be obtained using the best_params function.

# In[24]:


# Call the 'fit' method of rcv and pass the train data to it
rcv.fit(X.iloc[:split], y.iloc[:split])

# Call the 'best_params_' method to obtain the best parameters of C
best_C = rcv.best_params_["svc__C"]

# Call the 'best_params_' method to obtain the best parameters of kernel
best_kernel = rcv.best_params_["svc__kernel"]

# Call the 'best_params_' method to obtain the best parameters of gamma
best_gamma = rcv.best_params_["svc__gamma"]


# In[25]:


best_C


# In[26]:


best_kernel


# In[27]:


best_gamma


# ## Create the Support Vector Machine

# In this line of code we instantiate a new support vector classifier function with the best hyperparameters.

# In[28]:


# Create a new SVC classifier
cls = SVC(C=best_C, kernel=best_kernel, gamma=best_gamma)


# ### Train the data
#
# As done previously, for finding the best hyperparameters, you will first scale the data before you fit it to the classifier to train on. To do this, you need to first instantiate the Standard Scaler function.

# In[29]:


# Instantiate the StandardScaler
ss1 = StandardScaler()


# You will use the scaled training data to train the classifier algorithm.

# In[30]:


# Pass the scaled train data to the SVC classifier
# please note we are doing out-of-time (prediction) on unseen data
cls.fit(ss1.fit_transform(X.iloc[:split]), y.iloc[:split])


# ## Predict the signals

# Now, you can use the test data to make predictions and save the value of output 'y' in a list called 'y_predict'. This list will have the predicted values of 'Signal' for the test data.

# In[31]:


# Pass the test data to the predict function and store the values into 'y_predict'
y_predict = cls.predict(ss1.transform(X.iloc[split:]))


# Now create a new column 'Pred_Signal' in 'Df' to save all the predictions for both train data and test data.

# In[32]:


# Initiate a column by name, 'Pred_Signal' and assign 0 to it
Df["Pred_Signal"] = 0


# ### Save the predictions
# 1. To save predicted 'y' values of test data, we can simply assign 'y_predict' to 'Pred_Signal' using the split.
# 2. To save predicted 'y' values for train data, we make predictions for train data and save it similarly.

# In[33]:


# Save the predicted values for the train data
Df.iloc[:split, Df.columns.get_loc("Pred_Signal")] = pd.Series(
    cls.predict(ss1.transform(X.iloc[:split])).tolist()
)

# Save the predicted values for the test data
Df.iloc[split:, Df.columns.get_loc("Pred_Signal")] = y_predict


# Since, the algorithm was trained on the train data, it’s accuracy of prediction is expected to be better on this train data compared to the test data. You can print these two seperately to check the accuracies. (TRY ON YOUR OWN!)

# ## Use the model for trading strategy

# ### Trading strategy
#
# Our trading strategy is simply to buy/sell/do-nothing at that period for which the Signal is generated by the algorithm. The strategy assumes that you always get a fill at the "Open" prices.
#
# You had already calculated and saved returns on 'Open' prices in 'Ret'. You will create a column named 'Ret1' to store the strategy's returns based on the Signal.

# In[34]:


# Calculate strategy returns and store them in 'Ret1' column
Df["Ret1"] = Df["Ret"] * Df["Pred_Signal"]


# ## Analyze the performance

# Please note that here you are using only the test data to compare the performance of the strategy. You can pass the entire 'Ret1' column and check the performance on both the test and train data if you wish to.

# In[35]:


# Calculate the annualized Sharpe ratio
sharpe = np.sqrt(252) * Df["Ret1"][split:].mean() / Df["Ret1"][split:].std()

print("Sharpe", sharpe)


# ## Plot the results
#
# Now you can plot the results to visualize the performance.
#

# In[36]:


Df.set_index("Time", inplace=True)
# Plot the stretegy returns
plt.figure(figsize=(10, 5))
plt.plot(((Df["Ret"][split:] + 1).cumprod()), color="r", label="Market Returns")
plt.plot(((Df["Ret1"][split:] + 1).cumprod()), color="g", label="Strategy Returns")
plt.legend()
plt.show()


# # References
# <hr style="border:2px solid black"> </hr>

# - https://github.com/Datatouille/findalpha/tree/master

# In[ ]:
