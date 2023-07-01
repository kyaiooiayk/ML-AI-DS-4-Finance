#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Get-the-data" data-toc-modified-id="Get-the-data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Get the data</a></span></li><li><span><a href="#RNN" data-toc-modified-id="RNN-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>RNN</a></span></li><li><span><a href="#LSTM" data-toc-modified-id="LSTM-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>LSTM</a></span></li><li><span><a href="#References" data-toc-modified-id="References-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Modelling stock-prices with RNN and LSTM in TF
# 
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[1]:


import numpy as np
import pandas as pd
import math
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Dense, Dropout, 
                                     Activation, Flatten, 
                                     MaxPooling2D, SimpleRNN)
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


# In[ ]:





# # Get the data
# <hr style = "border:2px solid black" ></hr>

# In[2]:


def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# In[3]:


ticker = ['AAPL', 'MSFT']
start = datetime.datetime(2019, 1, 1)
end = datetime.datetime(2020, 1 ,1)
stock_prices = yf.download(ticker,start=start, end = end, interval='1d')               .Close


# In[4]:


diff_stock_prices = stock_prices.diff().dropna()


# In[5]:


split = int(len(diff_stock_prices['AAPL'].values) * 0.95)
diff_train_aapl = diff_stock_prices['AAPL'].iloc[:split]
diff_test_aapl = diff_stock_prices['AAPL'].iloc[split:]
diff_train_msft = diff_stock_prices['MSFT'].iloc[:split]
diff_test_msft = diff_stock_prices['MSFT'].iloc[split:]


# In[7]:


n_steps = 13
n_features = 1


# In[8]:


X_aapl, y_aapl = split_sequence(diff_train_aapl, n_steps)
X_aapl = X_aapl.reshape((X_aapl.shape[0],  X_aapl.shape[1],
                         n_features))


# # RNN
# <hr style = "border:2px solid black" ></hr>

# In[9]:


model = Sequential()
model.add(SimpleRNN(512, activation='relu', 
                    input_shape=(n_steps, n_features),
                    return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(256, activation = 'relu'))
model.add(Flatten())
model.add(Dense(1, activation='linear'))


# In[10]:


model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['mse'])


# In[11]:


history = model.fit(X_aapl, y_aapl, 
                    epochs=400, batch_size=150, verbose=1, 
                    validation_split = 0.10)


# In[12]:


start = X_aapl[X_aapl.shape[0] - n_steps]
x_input = start
x_input = x_input.reshape((1, n_steps, n_features))


# In[13]:


tempList_aapl = []
for i in range(len(diff_test_aapl)):
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    x_input = np.append(x_input, yhat)
    x_input = x_input[1:]
    tempList_aapl.append(yhat)


# In[14]:


X_msft, y_msft = split_sequence(diff_train_msft, n_steps)
X_msft = X_msft.reshape((X_msft.shape[0],  X_msft.shape[1],
                         n_features))


# In[15]:


history = model.fit(X_msft, y_msft, 
                    epochs=400, batch_size=150, verbose=1, 
                    validation_split = 0.10)


# In[16]:


start = X_msft[X_msft.shape[0] - n_steps]
x_input = start
x_input = x_input.reshape((1, n_steps, n_features))


# In[17]:


tempList_msft = []
for i in range(len(diff_test_msft)):
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    x_input = np.append(x_input, yhat)
    x_input = x_input[1:]
    tempList_msft.append(yhat)


# In[18]:


fig, ax = plt.subplots(2,1, figsize=(18,15))
ax[0].plot(diff_test_aapl, label='Actual Stock Price', linestyle='--')
ax[0].plot(diff_test_aapl.index, np.array(tempList_aapl).flatten(),
           linestyle='solid', label="Prediction")
ax[0].set_title('Predicted Stock Price-Apple')
ax[0].legend(loc='best')
ax[1].plot(diff_test_msft, label='Actual Stock Price', linestyle='--')
ax[1].plot(diff_test_msft.index,np.array(tempList_msft).flatten(),
           linestyle='solid', label="Prediction")
ax[1].set_title('Predicted Stock Price-Microsoft')
ax[1].legend(loc='best')


for ax in ax.flat:
    ax.set(xlabel='Date', ylabel='Differenced Price')
plt.show()


# # LSTM
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - LSTM tries to attack the weakness of RNN regarding long-term dependencies. 
# - LSTM has a quite useful tool to get rid of the unnecessary information so that it works more efficiently. 
# - LSTM works with gates, enabling it to forget irrelevant data. 
# - These gates are:
#     - Forget gates
#     - Input gates
#     - Output gates
# 
# </font>
# </div>

# In[20]:


n_steps = 13
n_features = 1


# In[21]:


model = Sequential()
model.add(LSTM(512, activation='relu',
          input_shape=(n_steps, n_features),
          return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256,activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='linear'))


# In[22]:


model.compile(optimizer='rmsprop', loss='mean_squared_error',
              metrics=['mse'])


# In[23]:


history = model.fit(X_aapl, y_aapl, 
                    epochs=400, batch_size=150, verbose=1, 
                    validation_split = 0.10)


# In[24]:


start = X_aapl[X_aapl.shape[0] - 13]
x_input = start
x_input = x_input.reshape((1, n_steps, n_features))


# In[25]:


tempList_aapl = []
for i in range(len(diff_test_aapl)):
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    x_input = np.append(x_input, yhat)
    x_input = x_input[1:]
    tempList_aapl.append(yhat)


# In[26]:


history = model.fit(X_msft, y_msft, 
                    epochs=400, batch_size=150, verbose=1, 
                    validation_split = 0.10)


# In[27]:


start = X_msft[X_msft.shape[0] - 13]
x_input = start
x_input = x_input.reshape((1, n_steps, n_features))


# In[28]:


tempList_msft = []
for i in range(len(diff_test_msft)):
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    x_input = np.append(x_input, yhat)
    x_input = x_input[1:]
    tempList_msft.append(yhat)


# In[29]:


fig, ax = plt.subplots(2, 1, figsize=(18, 15))
ax[0].plot(diff_test_aapl, label='Actual Stock Price', linestyle='--')
ax[0].plot(diff_test_aapl.index, np.array(tempList_aapl).flatten(),
           linestyle='solid', label="Prediction")
ax[0].set_title('Predicted Stock Price-Apple')
ax[0].legend(loc='best')
ax[1].plot(diff_test_msft, label='Actual Stock Price', linestyle='--')
ax[1].plot(diff_test_msft.index, np.array(tempList_msft).flatten(),
           linestyle='solid', label="Prediction")
ax[1].set_title('Predicted Stock Price-Microsoft')
ax[1].legend(loc='best')

for ax in ax.flat:
    ax.set(xlabel='Date', ylabel='$')
plt.show()


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://github.com/abdullahkarasan/mlfrm/blob/main/codes/chp_3.ipynb
# - Machine Learning for Financial Risk Management with Python Abdullah Karasan
#     
# </font>
# </div>

# In[ ]:




