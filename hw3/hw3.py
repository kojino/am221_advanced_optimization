
# coding: utf-8

# In[1]:


from cvxopt import matrix, solvers
import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('http://rasmuskyng.com/am221_spring18/psets/hw3/wines.csv',delimiter=';')


# In[3]:


n = df.shape[0]
m = df.shape[1] - 1


# In[4]:


X = df.iloc[:,:-1].as_matrix()
One_2n = np.vstack((np.ones((n,1)),-1*np.ones((n,1))))
In = np.identity(n)
A = matrix(np.hstack((np.vstack((X,-X)),One_2n,np.vstack((-In,-In)))))


# In[5]:


y = df.iloc[:,-1].as_matrix().astype(float)
b = matrix(np.hstack((y,-y)))


# In[6]:


c = matrix(np.hstack((np.zeros(m+1),np.repeat(1/n,n))))


# In[7]:


d=solvers.lp(c,A,b)

