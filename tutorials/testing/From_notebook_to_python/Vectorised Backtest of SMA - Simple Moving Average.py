#!/usr/bin/env python
# coding: utf-8

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
#
# **What?** Vectorised Backtest of SMA - Simple Moving Average
#
# </font>
# </div>

# # Vectorised vs. event-driven backtesting
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# - A **vectorised** backtest is the most basic way to evaluate a strategy. It simply multiplies a signal vector that represents the target position size with a vector of returns for the investment horizon to compute the period performance (some used the term equity curve). Vectorised backtesters are for quick research ideas. vectorized backtesting with NumPy and pandas is generally convenient and efficient to implement due to the concise code, and it is fast to execute due to these packages being optimized for such operations.
# - An **event-driven** backtester is a more well thought out simulation. By making use of an event driven backtester we can stop look ahead bias to a large extent by only feeding in the data as it becomes available. This also very closely matches how your trading will take place in real life via an execution system. We also have the advantage of building in transaction costs, liquidity constraints, and market impact. This is not something you can do with the vectorized method. (You could add transaction costs after the fact).
# - There seems to be a bit of a confusion of what what vectorised really means: if behind the scenes, the said vectorised function is simply passing the elements through a loop, just like an event-driven backtester in that sense. Then, there's no correctness reason for preferring one approach over another.
#
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[26]:


import yfinance as yf
from itertools import product
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn")
from tqdm import tqdm


# In[2]:


# Getting rid of the warning messages
import warnings

warnings.filterwarnings("ignore")

# Pandas future warning
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas


# # Get the data
# <hr style = "border:2px solid black" ></hr>

# In[3]:


# pull historical data
def get_data(pair):
    NUM_DAYS = 10000  # The number of days of historical data to retrieve
    INTERVAL = "1d"
    symbol = str(pair)  # Symbol of the desired stock

    # define start & dates
    start = datetime.date.today() - datetime.timedelta(NUM_DAYS)
    end = datetime.datetime.today()

    # pull data
    df = yf.download(symbol, start=start, end=end, interval=INTERVAL)
    return df


# In[4]:


pair = "AUDUSD=X"
data = get_data(pair)


# # SMA - Simple Moving Average
# <hr style = "border:2px solid black" ></hr>

# In[5]:


# No of day you want to consider
sma_short = 55
sma_long = 200

# Calculate SMA values
data["SMA_S"] = data.Close.rolling(sma_short).mean()
data["SMA_L"] = data.Close.rolling(sma_long).mean()


# Drop missing values
data = data.dropna()
data


# In[6]:


data.plot(
    figsize=(18, 7),
    y=["Close", "SMA_S", "SMA_L"],
    title=f"{pair} + SMA{sma_short} & SMA{sma_long}",
)
# plt.grid()
plt.show()


# # SMA strategy logic
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# - Where short SMA is greater than the long SMA, the position is +1 (buy)
# - Where short SMA is less than the long SMA, the position is -1 (sell).
#
# </font>
# </div>

# In[7]:


# Define Positions (Long or Short). Where 1 = Buy, -1 = Sell
data["position"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1)
data


# # Define Buy & Hold Returns
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# - The daily log returns are calculated by taking the natural logarithm of the division of the present close price by the previous price.
# - Log returns are an important financial metrics for determining the returns of an investment.
# - Strategy returns is the return from the position taken based on the previous close price. i.e the return for the present close price multiplied by the signal position — Buy(+1) or Sell(-1). The logic behind this is simple; if by the end of day 1, the signal says “Buy (+1)” and you take a long position, by day 2, the return of the strategy from the position taken from the signal on day 1 would be the log returns between the price of day 1 and day 2 multiplied by the position itself. E.g. If the position at day 1 is -1 and the log returns at day 2 is -0.00302, the strategy returns = -1 * -0.00302 i.e a profit return of 0.00302
#
# </font>
# </div>

# In[8]:


data.Close / data.Close.shift(1)


# In[9]:


data.Close.div(data.Close.shift(1))


# In[10]:


data["returns"] = np.log(data.Close.div(data.Close.shift(1)))


# daily return of strategy
data["strategy"] = data.position.shift(1) * data["returns"]
data.dropna(inplace=True)


# In[11]:


data.head()


# # Calculate Absolute Performance
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# - **Absolute performance** — the sum of the log returns of buy & hold and strategy. This gives an overview of the absolute returns after the test period.
# - **Annual returns** — The mean of the log returns of buy & hold and strategy multiplied by the total number of trading days in a year (252).
# - **Annual risk** — The standard deviation of the log returns of buy & hold and strategy multiplied by the square root of total number of trading days in a year (252).
# - **Cumulative Returns** — is calculated by the cumulative sum of exponential of the strategy log returns and the buy & hold returns
#
# </font>
# </div>

# In[12]:


# calculate absolute performance
data[["returns", "strategy"]].sum()


# In[13]:


# CALCULATE ACTUAL VALUE OF ABSOLUTE PERFORMANCE
data[["returns", "strategy"]].sum().apply(np.exp)


# In[14]:


# annual return
data[["returns", "strategy"]].mean() * 252


# In[15]:


# annual risk
data[["returns", "strategy"]].std() * np.sqrt(252)


# In[16]:


data["cum_returns"] = data["returns"].cumsum().apply(np.exp)
data["cum_strategy"] = data["strategy"].cumsum().apply(np.exp)
data


# In[17]:


# Visualize Cummulative returns
data[["cum_returns", "cum_strategy"]].plot(
    figsize=(20, 7), title=f"{pair} Returns + SMA{sma_short} & SMA{sma_long}"
)
plt.legend(fontsize=12)
plt.show()


# <div class="alert alert-info">
# <font color=black>
#
# - This shows the strategy using a SMA(sma_short = 55, sma_long = 200) performed worse than a buy & hold investment and would not be profitable to trade.
# - The strategy performance can be calculated by subtracting the last cumulative returns of buy & hold from the last cumulative return of the strategy. T0his shows a -11.0% performance compared to buy & hold
#
# </font>
# </div>

# In[18]:


strat_perf = data.cum_strategy.iloc[-1] - data.cum_returns.iloc[-1]
round(strat_perf, 5) * 100


# # Optimise SMA combination
# <hr style = "border:2px solid black" ></hr>

# In[20]:


# Create Backtest function
def sma_backtest(data, sma_short, sma_long):
    df = data.copy()
    df["returns"] = np.log(df.Close.div(df.Close.shift(1)))
    df["SMA_S"] = data.Close.rolling(sma_short).mean()
    df["SMA_L"] = data.Close.rolling(sma_long).mean()
    df.dropna(inplace=True)

    df["position"] = np.where(df["SMA_S"] > df["SMA_L"], 1, -1)
    df["strategy"] = df.position.shift(1) * df["returns"]
    df.dropna(inplace=True)

    # returns the absolute performance of strateg
    return round(np.exp(df["strategy"].sum()), 5)


# In[21]:


# range of sma
range_sma_short = range(3, 50, 1)
range_sma_long = range(55, 250, 1)


# In[22]:


# combination of SMAs
smas = list(product(range_sma_short, range_sma_long))
smas


# In[23]:


len(smas)


# In[27]:


# iterate through sma combination and test strategy
abs_perform = []
for pro in tqdm(smas):
    abs_perform.append(sma_backtest(data, pro[0], pro[1]))


# In[28]:


# Get max value
np.max(abs_perform)


# In[32]:


# results to dataframe
results = pd.DataFrame(smas, columns=["SMA_short", "SMA_long"])
results["abs_performance"] = abs_perform
results.sort_values("abs_performance", ascending=False)


# In[34]:


# Get max value combination
best_comb = smas[np.argmax(abs_perform)]
best_comb


# In[35]:


# top 10 best performing combination
best = results.nlargest(10, "abs_performance")
best.head()


# In[36]:


# top 10 least performing combination
least = results.nsmallest(10, "abs_performance")
least.head()


# # Check best performing combination
# <hr style = "border:2px solid black" ></hr>

# In[37]:


# define SMA values
sma_short = best_comb[0]
sma_long = best_comb[1]

# calculate SMA values
new_data = data.Close.to_frame()
new_data["SMA_s"] = data.Close.rolling(sma_short).mean()
new_data["SMA_l"] = data.Close.rolling(sma_long).mean()

# drop missing values
new_data = new_data.dropna()
new_data


# In[38]:


new_data["2021"].plot(figsize=(18, 7), title=f"{pair} + SMA{sma_short} & SMA{sma_long}")
plt.show()


# In[39]:


# Define Positions (Long or Short). Where 1 = Buy, -1 = Sell
new_data["position"] = np.where(new_data["SMA_s"] > new_data["SMA_l"], 1, -1)
new_data


# In[40]:


new_data["returns"] = np.log(new_data.Close.div(new_data.Close.shift(1)))


# daily return of strategy
new_data["strategy"] = new_data.position.shift(1) * new_data["returns"]
new_data.dropna(inplace=True)
new_data


# In[41]:


# calculate absolute performance
new_data[["returns", "strategy"]].sum()


# In[42]:


# CALCULATE ACTUAL VALUE OF ABSOLUTE PERFORMANCE
new_data[["returns", "strategy"]].sum().apply(np.exp)


# In[43]:


# annual return
new_data[["returns", "strategy"]].mean() * 252


# In[44]:


# annual risk
new_data[["returns", "strategy"]].std() * np.sqrt(252)


# In[45]:


new_data["cum_returns"] = new_data["returns"].cumsum().apply(np.exp)
new_data["cum_strategy"] = new_data["strategy"].cumsum().apply(np.exp)
new_data


# In[46]:


# Visualize Cummulative returns
new_data[["cum_returns", "cum_strategy"]].plot(
    figsize=(20, 7), title=f"{pair} Returns + SMA{sma_short} & SMA{sma_long}"
)
plt.legend(fontsize=12)
plt.show()


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
#
# - [YAHOO AUD/USD](https://finance.yahoo.com/quote/AUDUSD=X/?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvLnVrLw&guce_referrer_sig=AQAAALNbKvdnnyycC3agGxv2fNzx__G73WarjVfBU63B7R9XGoqbvxhA3w9GcR3L9FLa8dLwc66QMCF26cabsJGtchOasiKqmt8cQaoNpVmRM6nZyYSu0yh96F93qHmzud_O1HgswybhoZM4CTiX9kybecUul4nmAlvBODyHEXBmyJ5t)
# - [Blog article](https://wire.insiderfinance.io/vectorized-backtest-of-sma-cross-strategy-991cb13ffcd0)
# - [Why do we need event-driven backtesters?](https://quant.stackexchange.com/questions/46791/why-do-we-need-event-driven-backtesters)
# - [Why log return?](https://quantivity.wordpress.com/2011/02/21/why-log-returns/)
#
# </font>
# </div>

# In[ ]:
