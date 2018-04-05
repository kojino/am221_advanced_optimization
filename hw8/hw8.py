
# coding: utf-8

# In[15]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import linalg as LA
from sklearn import datasets


# In[16]:


# load data
df = pd.read_csv('http://rasmuskyng.com/am221_spring18/psets/hw7/banknotes.data',header=None)


# In[17]:


# convert labels
df[4] = 2 * df[4] - 1

# get features and labels
x_np = np.array(df.iloc[:,:4])
y_np = np.expand_dims(np.array(df[4]),axis=1)
n,d = x_np.shape

# hyper parameters
lam = 50
t = 1


# In[18]:


import torch
from torch.autograd import Variable


# In[19]:


x = Variable(torch.DoubleTensor(x_np),requires_grad=False)
y = Variable(torch.DoubleTensor(y_np),requires_grad=False)


# In[20]:


def gradient_descent(w_b_xi, lr, t):
    # define variables for gradient descent
    w_np, b_np, xi_np = w_b_xi[:d], w_b_xi[d], w_b_xi[d+1:]
    w = Variable(torch.DoubleTensor(w_np),requires_grad=True)
    b = Variable(torch.DoubleTensor(b_np),requires_grad=True)
    xi = Variable(torch.DoubleTensor(xi_np),requires_grad=True)
    
    # run gradient descent
    for i in range(10000):
        # calculate barrier objective
        barrier_objective = t * (0.5 * w.norm() ** 2 + lam * torch.sum(xi))                       - torch.sum(torch.log(y * x @ w + b + xi - 1))                       - torch.sum(torch.log(xi))
        # run back propagation
        barrier_objective.backward()
        # update the parameters
        w.data -= lr * w.grad.data
        b.data -= lr * b.grad.data
        xi.data -= lr * xi.grad.data

        # manually zero the gradients after running the backward pass
        w.grad.data.zero_()
        b.grad.data.zero_()
        xi.grad.data.zero_()
        
    return np.vstack([w.data.numpy(),np.expand_dims(b.data.numpy(),1),xi.data.numpy()])


# In[21]:


def f(X,lam):
    # calculate the true objective
    w, b, xi = X[:d], X[d], X[d+1:]
    return 1/2 * np.linalg.norm(w, ord=2) ** 2 + lam * np.sum(xi)


# In[22]:


def acc(X,x,y):
    # calculate the accuracy
    w, b, xi = X[:d], X[d], X[d+1:]
    return np.mean(y * (x @ w + b) >= 1)


# In[28]:


# initial feasible solution
w_b_xi = np.expand_dims(np.hstack([0*np.ones(d+1),np.repeat(2,n)]),axis=1)

# more hyper parameters
eps = 0.0001
mu = 1.1
t = 1

# barrier method
accs = []
mus = [1.1, 1.5, 2, 5, 10]
for mu in mus:
    print(mu)
    for i in range(25):
        w_b_xi = gradient_descent(w_b_xi,10**-8/t,t)
        print(i)
        print("t: {} | objective: {} | classification accuracy: {}".format(t,f(w_b_xi,lam),acc(w_b_xi,x_np,y_np)))
        t = mu * t
    accs.append(acc(w_b_xi,x_np,y_np))

