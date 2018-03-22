
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd

df = pd.read_csv('http://rasmuskyng.com/am221_spring18/psets/hw3/wines.csv',delimiter=';')


# In[2]:

n_dim = df.shape[0]
d_dim = df.shape[1]

X = np.hstack((df.iloc[:,:-1].as_matrix(),np.ones((n_dim,1))))
y = df.iloc[:,-1].as_matrix().astype(float)


# In[3]:

# a.
def is_psd(x):
    return np.all(np.linalg.eigvals(x) >= 0)
hessian = X.T @ X # no division by 2 for simplicity
hessian_evalues = np.linalg.eigvals(hessian)
M = max(hessian_evalues)
m = min(hessian_evalues)
I = np.identity(X.shape[1])
print(is_psd(M*I-hessian))
print(is_psd(hessian-m*I))


# In[4]:

# b.
d_closed = np.linalg.inv(X.T @ X) @ X.T @ y 


# In[5]:

def f(d):
    return np.linalg.norm(X @ d - y) ** 2 / n_dim


# In[6]:

a_closed,b_closed = d_closed[:-1], d_closed[-1]
val_closed = f(d_closed)
print("a:",a_closed,"\nb:",b_closed,"\nval:",val_closed)


# In[7]:

# c.
num_epoch = 100000
d_iter = np.hstack([np.random.normal(loc=0,scale=0.00001,size=d_dim-1),[1]])
d_init = d_iter.copy()
for i in range(num_epoch):
    delta = np.random.normal(loc=0,scale=0.00001,size=d_dim) 
    lam   = ((X@delta).T @ (y-X@d_iter))/np.linalg.norm(X@delta)**2
    d_iter = d_iter + lam * delta


# In[8]:

a_iter,b_iter = d_iter[:-1], d_iter[-1]
val_iter = f(d_iter)
print("a:",a_iter,"\nb:",b_iter,"\nval:",val_iter)


# In[9]:

# d.
(1-m/M)**num_epoch * (f(np.hstack([np.random.normal(loc=0,scale=0.00001,size=d_dim-1),[1]]))-f(d_closed))

