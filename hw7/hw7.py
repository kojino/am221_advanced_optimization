
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import linalg as LA
from sklearn import datasets


# In[3]:


df = pd.read_csv('http://rasmuskyng.com/am221_spring18/psets/hw7/banknotes.data',header=None)


# In[4]:


df[4] = 2 * df[4] - 1


# In[5]:


X = df.iloc[:,:4].copy()
y = df.iloc[:,4].copy()
target = y==-1


# In[8]:


# a. 
def perceptron(X):
    
    w = [0.0 for i in range(X.shape[1])]
    separated = False
    while not separated:
        separated = True
        for x in X:
            if w @ x <= 0:
                w += x/LA.norm(x, 2)
                separated=False

    return w



# In[7]:


# append -1 to the end of all rows
X = np.hstack([X,-np.ones((X.shape[0],1))])

# negate values in label 0
X[target] = -1 * X[target]

perceptron(X)


# In[18]:


# b.
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

X = df.iloc[:,:4].copy()
y = np.expand_dims(df.iloc[:,4].copy(),axis=-1)

n,d = X.shape
lambdas = [10.0**i for i in np.arange(-7,7,1)]


# In[19]:


def svm(lam):
    Q = matrix(np.vstack([np.hstack([np.eye(d),np.zeros([d,1+n])]),np.zeros([1+n,d+1+n])]))
    p = matrix(np.vstack([np.zeros([d+1,1]),lam * np.ones([n,1])]))
    h = matrix(np.vstack([-1 * np.ones([n,1]),np.zeros([n,1])]))
    G = matrix(np.vstack([np.hstack([-1 * y * X, -1 * y, -1 * np.ones([n,n])]),np.hstack([np.zeros([n,d+1]),-1 * np.eye(n)])]))
    sol=solvers.qp(Q, p, G, h)
    
    w_opt = np.array(sol['x'])[:d]
    b = np.array(sol['x'])[d]
    print(w_opt,b)
    acc = np.mean((X @ w_opt + b > 0).astype(float) == (y > 0).astype(float))
    return acc


# In[20]:


accuracies = []
for lam in lambdas:
    acc = svm(lam)
    accuracies.append(acc)
    print(lam,acc)


# In[22]:


import matplotlib.pyplot as plt
plt.plot([i for i in np.arange(-7,7,1)],accuracies)
plt.xlabel('lambda')
plt.ylabel('classification accuracy')
plt.show()

