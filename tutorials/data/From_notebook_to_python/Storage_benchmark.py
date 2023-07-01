#!/usr/bin/env python
# coding: utf-8

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Storage Benchmark
# 
# </font>
# </div>

# # Available storage formats
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-info">
# <font color=black>
# 
# 
# - **CSV**: Comma-separated, standard flat text file format.
# 
# - **HDF5**: Hierarchical data format, developed initially at the National Center for Supercomputing Applications. It is a fast and scalable storage format for numerical data, available in pandas using the PyTables library.
# 
# - **Parquet**: Part of the Apache Hadoop ecosystem, a binary, columnar storage format that provides efficient data compression and encoding and has been developed by Cloudera and Twitter. It is available for pandas through the `pyarrow` library.
# 
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[1]:


from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import string
import warnings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')


# ## Generate Test Data

# 
# <div class="alert alert-info">
# <font color=black>
# 
# - The test `DataFrame` that can be configured to contain numerical or text data, or both. 
# - Run this twice with the following settings:
#     - `data_type='Numeric: numerical_cols=2000, text_cols=0`
#     - `data_type='Mixed: numerical_cols=1000, text_cols=1000`
# 
#     
# </font>
# </div>

# In[2]:


results = {}


# In[3]:


def generate_test_data(nrows=100000, numerical_cols=2000, text_cols=0, text_length=10):
    s = "".join([random.choice(string.ascii_letters)
                 for _ in range(text_length)])
    data = pd.concat([pd.DataFrame(np.random.random(size=(nrows, numerical_cols))),
                      pd.DataFrame(np.full(shape=(nrows, text_cols), fill_value=s))],
                     axis=1, ignore_index=True)
    data.columns = [str(i) for i in data.columns]
    return data


# In[4]:


# Change these
data_type = 'Mixed'
df = generate_test_data(numerical_cols=2000, text_cols=1000)


# In[5]:


df.info()


# ## Parquet

# ### Size

# In[6]:


parquet_file = Path('test.parquet')


# In[7]:


df.to_parquet(parquet_file)
size = parquet_file.stat().st_size


# ### Read

# In[8]:


get_ipython().run_cell_magic('timeit', '-o', 'df = pd.read_parquet(parquet_file)')


# In[9]:


read = _


# In[10]:


parquet_file.unlink()


# ### Write

# In[11]:


get_ipython().run_cell_magic('timeit', '-o', 'df.to_parquet(parquet_file)\nparquet_file.unlink()')


# In[12]:


write = _


# ### Results

# In[13]:


results['Parquet'] = {'read': np.mean(read.all_runs), 'write': np.mean(write.all_runs), 'size': size}


# ## HDF5

# In[14]:


test_store = Path('index.h5')


# ### Fixed Format

# #### Size

# In[15]:


with pd.HDFStore(test_store) as store:
    store.put('file', df)
size = test_store.stat().st_size


# #### Read

# In[16]:


get_ipython().run_cell_magic('timeit', '-o', "with pd.HDFStore(test_store) as store:\n    store.get('file')")


# In[17]:


read = _


# In[18]:


test_store.unlink()


# #### Write

# In[19]:


get_ipython().run_cell_magic('timeit', '-o', "with pd.HDFStore(test_store) as store:\n    store.put('file', df)\ntest_store.unlink()")


# In[20]:


write = _


# #### Results

# In[21]:


results['HDF Fixed'] = {'read': np.mean(read.all_runs), 'write': np.mean(write.all_runs), 'size': size}


# ### Table Format

# #### Size

# In[22]:


with pd.HDFStore(test_store) as store:
    store.append('file', df, format='t')
size = test_store.stat().st_size    


# #### Read

# In[23]:


get_ipython().run_cell_magic('timeit', '-o', "with pd.HDFStore(test_store) as store:\n    df = store.get('file')")


# In[24]:


read = _


# In[25]:


test_store.unlink()


# #### Write

# Note that `write` in table format does not work with text data.

# In[26]:


get_ipython().run_cell_magic('timeit', '-o', "with pd.HDFStore(test_store) as store:\n    store.append('file', df, format='t')\ntest_store.unlink()    ")


# In[27]:


write = _


# #### Results

# In[28]:


results['HDF Table'] = {'read': np.mean(read.all_runs), 'write': np.mean(write.all_runs), 'size': size}


# ### Table Select

# #### Size

# In[29]:


with pd.HDFStore(test_store) as store:
    store.append('file', df, format='t', data_columns=['company', 'form'])
size = test_store.stat().st_size 


# #### Read

# In[30]:


company = 'APPLE INC'


# In[31]:


get_ipython().run_cell_magic('timeit', '', "with pd.HDFStore(test_store) as store:\n    s = store.get('file')")


# In[32]:


read = _


# In[33]:


test_store.unlink()


# #### Write

# In[34]:


get_ipython().run_cell_magic('timeit', '', "with pd.HDFStore(test_store) as store:\n    store.append('file', df, format='t', data_columns=['company', 'form'])\ntest_store.unlink() ")


# In[35]:


write = _


# #### Results

# In[36]:


results['HDF Select'] = {'read': np.mean(read.all_runs), 'write': np.mean(write.all_runs), 'size': size}


# ## CSV

# In[37]:


test_csv = Path('test.csv')


# ### Size

# In[38]:


df.to_csv(test_csv)
test_csv.stat().st_size


# ### Read

# In[39]:


get_ipython().run_cell_magic('timeit', '-o', 'df = pd.read_csv(test_csv)')


# In[40]:


read = _


# In[41]:


test_csv.unlink()  


# ### Write

# In[42]:


get_ipython().run_cell_magic('timeit', '-o', 'df.to_csv(test_csv)\ntest_csv.unlink()')


# In[43]:


write = _


# ### Results

# In[149]:


results['CSV'] = {'read': np.mean(read.all_runs), 'write': np.mean(write.all_runs), 'size': size}


# ## Store Results

# In[150]:


pd.DataFrame(results).assign(Data=data_type).to_csv(f'{data_type}.csv')


# ## Display Results

# Please run the notebook twice as described above under `Usage` to create the two `csv` files with results for different test data.

# In[160]:


df = (pd.read_csv('Numeric.csv', index_col=0)
      .append(pd.read_csv('Mixed.csv', index_col=0))
      .rename(columns=str.capitalize))
#df.index.name='Storage'
df = df.set_index('Data', append=True).unstack()
#df = df / 1e9


# In[161]:


df


# In[167]:


# Should need to use log scale for the storage
fig, axes = plt.subplots(ncols=5, figsize=(16, 4))
for i, method in enumerate(['Parquet', 'Hdf fixed', 'Hdf table', "Hdf select", "Csv"]):
    df.loc[:, method].plot.barh(title=method, ax=axes[i], logx=False)

fig.tight_layout()    


# In[165]:


# Should need to use log scale for the storage
fig, axes = plt.subplots(ncols=4, figsize=(16, 4))
for i, method in enumerate(['Parquet', 'Hdf fixed', 'Hdf table', "Csv"]):
    df.loc[:, method].plot.barh(title=method, ax=axes[i], logx=True)

fig.tight_layout()    


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - [GitHub code](https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/02_market_and_fundamental_data/05_storage_benchmark/storage_benchmark.ipynb)
# 
# </font>
# </div>

# In[ ]:




