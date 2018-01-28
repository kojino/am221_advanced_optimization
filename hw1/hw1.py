
# coding: utf-8

# In[1]:

import csv
import pandas as pd


# In[2]:

# read log
logs = pd.read_csv('access.log',delimiter='\t',header=None)
logs.columns = ['time_stamp','ip']


# In[3]:

# get 10 top ips
top_ten_logs = logs.groupby('ip').count()['time_stamp'].nlargest(10)
top_ten_ips = top_ten_logs.index.tolist()


# In[4]:

# save to txt
with open('top_ten_ips.txt', 'w') as f:
    f.write( '\n'.join( top_ten_ips ) )

