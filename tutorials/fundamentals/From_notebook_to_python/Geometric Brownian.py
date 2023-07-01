#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Main-objective" data-toc-modified-id="Main-objective-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Main objective</a></span></li><li><span><a href="#Plotting" data-toc-modified-id="Plotting-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Plotting</a></span></li><li><span><a href="#Clean-up-repository" data-toc-modified-id="Clean-up-repository-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Clean-up repository</a></span></li><li><span><a href="#References" data-toc-modified-id="References-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# In[1]:


import os
import random
import string

import click
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import pandas as pd


# # Main objective
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
#  
# - Demonstrate how to generate multiple CSV files of synthetic daily stock pricing and volume data using the analytical solution to the Geometric Brownian Motion stochastic differential equation.
# 
# - The purposes is to examine the 'what if?' scenarios in systematic trading.
#     
# </font>
# </div>

# In[2]:


class GeometricBrownianMotionAssetSimulator:
    """
    This callable class will generate a daily
    open-high-low-close-volume (OHLCV) based DataFrame to simulate
    asset pricing paths with Geometric Brownian Motion for pricing
    and a Pareto distribution for volume.

    It will output the results to a CSV with a randomly generated
    ticker smbol.

    For now the tool is hardcoded to generate business day daily
    data between two dates, inclusive.

    Note that the pricing and volume data is completely uncorrelated,
    which is not likely to be the case in real asset paths.

    Parameters
    ----------
    start_date : `str`
        The starting date in YYYY-MM-DD format.
    end_date : `str`
        The ending date in YYYY-MM-DD format.
    output_dir : `str`
        The full path to the output directory for the CSV.
    symbol_length : `int`
        The length of the ticker symbol to use.
    init_price : `float`
        The initial price of the asset.
    mu : `float`
        The mean 'drift' of the asset.
    sigma : `float`
        The 'volatility' of the asset.
    pareto_shape : `float`
        The parameter used to govern the Pareto distribution
        shape for the generation of volume data.
    """

    def __init__(
        self,
        start_date,
        end_date,
        output_dir,
        symbol_length,
        init_price,
        mu,
        sigma,
        pareto_shape
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = output_dir
        self.symbol_length = symbol_length
        self.init_price = init_price
        self.mu = mu
        self.sigma = sigma
        self.pareto_shape = pareto_shape

    def _generate_random_symbol(self):
        """
        Generates a random ticker symbol string composed of
        uppercase ASCII characters to use in the CSV output filename.

        Returns
        -------
        `str`
            The random ticker string composed of uppercase letters.
        """
        return ''.join(
            random.choices(
                string.ascii_uppercase,
                k=self.symbol_length
            )
        )

    def _create_empty_frame(self):
        """
        Creates the empty Pandas DataFrame with a date column using
        business days between two dates. Each of the price/volume
        columns are set to zero.

        Returns
        -------
        `pd.DataFrame`
            The empty OHLCV DataFrame for subsequent population.
        """
        date_range = pd.date_range(
            self.start_date,
            self.end_date,
            freq='B'
        )

        zeros = pd.Series(np.zeros(len(date_range)))

        return pd.DataFrame(
            {
                'date': date_range,
                'open': zeros,
                'high': zeros,
                'low': zeros,
                'close': zeros,
                'volume': zeros
            }
        )[['date', 'open', 'high', 'low', 'close', 'volume']]

    def _create_geometric_brownian_motion(self, data):
        """
        Calculates an asset price path using the analytical solution
        to the Geometric Brownian Motion stochastic differential
        equation (SDE).

        This divides the usual timestep by four so that the pricing
        series is four times as long, to account for the need to have
        an open, high, low and close price for each day. These prices
        are subsequently correctly bounded in a further method.

        Parameters
        ----------
        data : `pd.DataFrame`
            The DataFrame needed to calculate length of the time series.

        Returns
        -------
        `np.ndarray`
            The asset price path (four times as long to include OHLC).
        """
        n = len(data)
        T = n / 252.0  # Business days in a year
        dt = T / (4.0 * n)  # 4.0 is needed as four prices per day are required

        # Vectorised implementation of asset path generation
        # including four prices per day, used to create OHLC
        asset_path = np.exp(
            (self.mu - self.sigma**2 / 2) * dt +
            self.sigma * np.random.normal(0, np.sqrt(dt), size=(4 * n))
        )

        return self.init_price * asset_path.cumprod()

    def _append_path_to_data(self, data, path):
        """
        Correctly accounts for the max/min calculations required
        to generate a correct high and low price for a particular
        day's pricing.

        The open price takes every fourth value, while the close
        price takes every fourth value offset by 3 (last value in
        every block of four).

        The high and low prices are calculated by taking the max
        (resp. min) of all four prices within a day and then
        adjusting these values as necessary.

        This is all carried out in place so the frame is not returned
        via the method.

        Parameters
        ----------
        data : `pd.DataFrame`
            The price/volume DataFrame to modify in place.
        path : `np.ndarray`
            The original NumPy array of the asset price path.
        """
        data['open'] = path[0::4]
        data['close'] = path[3::4]

        data['high'] = np.maximum(
            np.maximum(path[0::4], path[1::4]),
            np.maximum(path[2::4], path[3::4])
        )

        data['low'] = np.minimum(
            np.minimum(path[0::4], path[1::4]),
            np.minimum(path[2::4], path[3::4])
        )

    def _append_volume_to_data(self, data):
        """
        Utilises a Pareto distribution to simulate non-negative
        volume data. Note that this is not correlated to the
        underlying asset price, as would likely be the case for
        real data, but it is a reasonably effective approximation.

        Parameters
        ----------
        data : `pd.DataFrame`
            The DataFrame to append volume data to, in place.
        """
        data['volume'] = np.array(
            list(
                map(
                    int,
                    np.random.pareto(
                        self.pareto_shape,
                        size=len(data)
                    ) * 1000000.0
                )
            )
        )

    def _output_frame_to_dir(self, symbol, data):
        """
        Output the fully-populated DataFrame to disk into the
        desired output directory, ensuring to trim all pricing
        values to two decimal places.

        Parameters
        ----------
        symbol : `str`
            The ticker symbol to name the file with.
        data : `pd.DataFrame`
            The DataFrame containing the generated OHLCV data.
        """
        output_file = os.path.join(self.output_dir, '%s.csv' % symbol)
        data.to_csv(output_file, index=False, float_format='%.2f')

    def __call__(self):
        """
        The entrypoint for generating the asset OHLCV frame. Firstly this
        generates a symbol and an empty frame. It then populates this
        frame with some simulated GBM data. The asset volume is then appended
        to this data and finally it is saved to disk as a CSV.
        """
        symbol = self._generate_random_symbol()
        data = self._create_empty_frame()
        path = self._create_geometric_brownian_motion(data)
        self._append_path_to_data(data, path)
        self._append_volume_to_data(data)
        self._output_frame_to_dir(symbol, data)


# In[3]:


num_assets = 5
random_seed = 41
start_date = '1993-01-01'
end_date = '2022-07-31'
output_dir = './'
symbol_length = 5
init_price = 100.0
mu = 0.1
sigma = 0.3
pareto_shape = 1.5


# In[4]:


num_assets = int(num_assets)
random_seed = int(random_seed)
symbol_length = int(symbol_length)
init_price = float(init_price)
mu = float(mu)
sigma = float(sigma)
pareto_shape = float(pareto_shape)

# Need to seed both Python and NumPy separately
random.seed(random_seed)
np.random.seed(seed=random_seed)

gbmas = GeometricBrownianMotionAssetSimulator(
    start_date,
    end_date,
    output_dir,
    symbol_length,
    init_price,
    mu,
    sigma,
    pareto_shape
)

# Create num_assets files by repeatedly calling
# the instantiated class
for i in range(num_assets):
    print('Generating asset path %d of %d...' % (i+1, num_assets))
    gbmas()



# In[5]:


get_ipython().system('ls ./*csv')


# # Plotting
# <hr style = "border:2px solid black" ></hr>

# In[6]:


# Change BUAWV.csv to your preferred ticker symbol
df = pd.read_csv('./BUAWV.csv').set_index('date')
df[['open', 'high', 'low', 'close']].plot()
plt.show()


# # Clean-up repository
# <hr style = "border:2px solid black" ></hr>

# In[7]:


get_ipython().system('rm ./DSERQ.csv')
get_ipython().system('rm ./JFEXP.csv')
get_ipython().system('rm ./ROJTO.csv')
get_ipython().system('rm ./BUAWV.csv ')
get_ipython().system('rm ./URVLE.csv')


# In[8]:


get_ipython().system('ls ./*csv')


# # References
# <hr style = "border:2px solid black" ></hr>

# 
# - https://www.quantstart.com/articles/geometric-brownian-motion-simulation-with-python/

# In[ ]:




