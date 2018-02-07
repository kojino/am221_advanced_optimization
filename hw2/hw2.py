
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from sklearn import datasets

iris = datasets.load_iris()
data = iris.data
target = iris.target==0


# In[2]:

# Plot the training points
plt.scatter(data[:, 0], data[:, 1], c=target)
plt.savefig("iris.png")
plt.show()


# In[3]:

def train(X):
    w = [0.0 for i in range(X.shape[1])]
    separated = False
    while not separated:
        separated = True
        for x in X:
            if w @ x <= 0:
                w += x/LA.norm(x, 2)
                separated=False

    return w


# In[4]:

X = data.copy()
# append -1 to the end of all rows
X = np.hstack([X,-np.ones((X.shape[0],1))])


# In[5]:

# negate values in label 0
X[target] = -1 * X[target]


# In[6]:

w = train(X)


# In[7]:

w


# In[ ]:



