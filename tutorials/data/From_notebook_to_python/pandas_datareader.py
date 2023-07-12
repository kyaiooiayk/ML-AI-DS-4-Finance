#!/usr/bin/env python
# coding: utf-8

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** YAHOO finacial data API
# 
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[ ]:


import os
import pandas_datareader as pdr


# # Tiingo
# <hr style = "border:2px solid black" ></hr>

# - required you to registred
# - https://www.tiingo.com/

# In[ ]:


df = pdr.get_data_tiingo('GOOG', api_key=os.getenv('TIINGO_API_KEY'))
df.head()


# # IEX
# <hr style = "border:2px solid black" ></hr>

# # Alpha vantage
# <hr style = "border:2px solid black" ></hr>

# # Forex
# <hr style = "border:2px solid black" ></hr>

# # Econdb
# <hr style = "border:2px solid black" ></hr>

# # Enigma
# <hr style = "border:2px solid black" ></hr>

# # Quandl
# <hr style = "border:2px solid black" ></hr>

# # FRED
# <hr style = "border:2px solid black" ></hr>

# # Fama/French
# <hr style = "border:2px solid black" ></hr>

# # World Bank
# <hr style = "border:2px solid black" ></hr>

# # OECD
# <hr style = "border:2px solid black" ></hr>

# # Eurostat
# <hr style = "border:2px solid black" ></hr>

# # TSP
# <hr style = "border:2px solid black" ></hr>

# # Nasdaq
# <hr style = "border:2px solid black" ></hr>

# In[1]:


from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
symbols = get_nasdaq_symbols()
print(symbols.loc['IBM'])


# In[2]:


import pandas_datareader.data as web
industrial_production = web.DataReader(
    'IPGMFN', 'fred', '1988', '2017-12').squeeze().dropna()
nasdaq = web.DataReader('NASDAQCOM', 'fred', '1990',
                        '2017-12-31').squeeze().dropna()


# In[3]:


nasdaq


# # Stooq
# <hr style = "border:2px solid black" ></hr>

# # MOEX
# <hr style = "border:2px solid black" ></hr>

# # Naver
# <hr style = "border:2px solid black" ></hr>

# # Yahoo
# <hr style = "border:2px solid black" ></hr>

# In[ ]:





# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - [How to use `pandas_datareader`](https://pandas-datareader.readthedocs.io/en/latest/remote_data.html)
# 
# </font>
# </div>

# In[ ]:




