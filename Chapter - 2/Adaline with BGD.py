#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 12:28:51 2017

@author: ashish
"""

class Adaline(object):
    
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        self.w_ = np.zeros( 1 + X.shape[1])
        self.cost_ = []
        # instead of iterating each training sample as in perceptron,
        # we calculate gradient descent on whole training dataset
        for i in range(self.n_iter):
            output = self.net_input(X)
            output = output.reshape(y.shape[0], y.shape[1])
            errors = y - output
            a = X.T.dot(errors)
            a = a.reshape(a.shape[1], a.shape[0])
            self.w_[1:] += self.eta * a[0]
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        
        return self
        
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        return self.net_input(X)
        
    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)
        
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

# load data same as in perceptron model        
iris = load_iris()
X = pd.DataFrame(iris.data)
X = X.iloc[0:100, [0, 2]].values
y = pd.DataFrame(iris.target)
y = y.iloc[0:100].values
y = np.where(y==0, -1, 1)

fig, p = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

# eta = 0.01
ada1 = Adaline(n_iter=10, eta=0.01).fit(X, y)
p1 = p[0]
p1.plot(range(1, len(ada1.cost_)+1), np.log10(ada1.cost_), marker='o')
p1.set_xlabel('Epochs')
p1.set_ylabel('log(SSE)')
p1.set_title('Adaline with eta=0.01')

# eta = 0.0001
ada2 = Adaline(n_iter=10, eta=0.0001).fit(X, y)
p2 = p[1]
p2.plot(range(1, len(ada2.cost_)+1), np.log10(ada2.cost_), marker='o')
p2.set_xlabel('Epochs')
p2.set_ylabel('log(SSE)')
p2.set_title('Adaline with eta=0.0001')