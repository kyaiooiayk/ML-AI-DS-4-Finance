#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Import-data" data-toc-modified-id="Import-data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Import data</a></span></li><li><span><a href="#Annualised-Sharpe-ratio" data-toc-modified-id="Annualised-Sharpe-ratio-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Annualised Sharpe ratio</a></span></li><li><span><a href="#Equity-Sharpe-ratio" data-toc-modified-id="Equity-Sharpe-ratio-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Equity Sharpe ratio</a></span></li><li><span><a href="#Maket-neutral-Sharpe-ratio" data-toc-modified-id="Maket-neutral-Sharpe-ratio-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Maket neutral Sharpe ratio</a></span></li><li><span><a href="#References" data-toc-modified-id="References-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
#
# **What?** Sharpe Ratio
#
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[8]:


import datetime
import numpy as np
import pandas as pd
import urllib
import pandas_datareader as pdr
import datetime as dt


# # Import data
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# - Where our portfolio will consist of the tickers for Apple, Microsoft, Twitter and IBM (AAPL, MSFT, TWTR, IBM). We read the data from start 2020 from the Yahoo! Finance API using Pandas Datareader.
#
# - Finally, we only keep the Adjusted Close price.
#
# </font>
# </div>

# In[9]:


def get_historic_data(ticker, start_date, end_date):
    """
    Obtains data from Yahoo Finance and adds it to a pandas DataFrame object.

    ticker: Yahoo Finance ticker symbol, e.g. "GOOG" for Google, Inc.
    start_date: Start date in (YYYY, M, D) format
    end_date: End date in (YYYY, M, D) format
    """

    pdf = pdr.get_data_yahoo("AAPL", start, end)

    return pdf


# # Annualised Sharpe ratio
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# - Any strategy could have a periods of mediocre returns and extensive drawdown. Thus a major challenge for quant researchers lies in identifying when a strategy is **truly underperforming** due to erosion of edge or whether it is a "temporary" period of poorer performance.
# - This motivates the need for an effective **trailing metric** that captures current performance of the strategy in relation to its previous performance.
# - One of the most widely used measures is the **annualised rolling Sharpe ratio**.
#
# </font>
# </div>

# In[10]:


def annualised_sharpe(returns, N=252):
    """
    Calculate the annualised Sharpe ratio of a returns stream
    based on a number of trading periods, N. N defaults to 252,
    which then assumes a stream of daily returns.

    The function assumes that the returns are the excess of
    those compared to a benchmark.
    """
    return np.sqrt(N) * returns.mean() / returns.std()


# # Equity Sharpe ratio
# <hr style = "border:2px solid black" ></hr>

# In[23]:


def equity_sharpe(ticker, start, end):
    """
    Calculates the annualised Sharpe ratio based on the daily
    returns of an equity ticker symbol listed in Yahoo Finance.

    The dates have been hardcoded here for the QuantStart article
    on Sharpe ratios.
    """

    # Obtain the equities daily historic data for the desired time period
    # and add to a pandas DataFrame

    pdf = get_historic_data(ticker, start, end)
    # Use the percentage change method to easily calculate daily returns
    pdf["daily_ret"] = pdf["Adj Close"].pct_change()

    # Assume an average annual risk-free rate over the period of 5%
    pdf["excess_daily_ret"] = pdf["daily_ret"] - 0.05 / 252

    # Return the annualised Sharpe ratio based on the excess daily returns
    return annualised_sharpe(pdf["excess_daily_ret"])


# In[24]:


start = dt.datetime(2000, 1, 1)
end = dt.datetime(2013, 1, 1)

equity_sharpe("GOOG", start, end)


# # Maket neutral Sharpe ratio
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#
# - Now we can try the same calculation for a market-neutral strategy. The goal of this strategy is to fully isolate a particular equity’s performance from the market in general.
#
# - The simplest way to achieve this is to go short an equal amount (in dollars) of an Exchange Traded Fund (ETF) that is designed to track such a market. The most obvious choice for the US large-cap equities market is the S&P500 index, which is tracked by the SPDR ETF, with the ticker of SPY.
#
# - To calculate the annualised Sharpe ratio of such a strategy we will obtain the historical prices for SPY and calculate the percentage returns in a similar manner to the previous stocks, with the exception that we will not use the risk-free benchmark. We will calculate the net daily returns which requires subtracting the difference between the long and the short returns and then dividing by 2, as we now have twice as much trading capital.
#
# </font>
# </div>

# In[28]:


def market_neutral_sharpe(ticker, benchmark, start, end):
    """
    Calculates the annualised Sharpe ratio of a market
    neutral long/short strategy inolving the long of 'ticker'
    with a corresponding short of the 'benchmark'.
    """

    # Get historic data for both a symbol/ticker and a benchmark ticker
    # The dates have been hardcoded, but you can modify them as you see fit!

    tick = get_historic_data(ticker, start, end)
    bench = get_historic_data(benchmark, start, end)

    # Calculate the percentage returns on each of the time series
    tick["daily_ret"] = tick["Adj Close"].pct_change()
    bench["daily_ret"] = bench["Adj Close"].pct_change()

    # Create a new DataFrame to store the strategy information
    # The net returns are (long - short)/2, since there is twice
    # trading capital for this strategy
    strat = pd.DataFrame(index=tick.index)
    strat["net_ret"] = (tick["daily_ret"] - bench["daily_ret"]) / 2.0

    # Return the annualised Sharpe ratio for this strategy
    return annualised_sharpe(strat["net_ret"])


# In[30]:


market_neutral_sharpe("GOOG", "SPY", start, end)


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
#
# - https://www.quantstart.com/articles/Sharpe-Ratio-for-Algorithmic-Trading-Performance-Measurement/
#
# </font>
# </div>

# In[ ]:
