#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 12:05:30 2017

@author: ashish
"""

class AdalineSGD(object):
    
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        # we shuffle data for better performance with Stochastic GD
        self.shuffle = shuffle
        self.w_initialized = False
        if random_state:
            seed(random_state)
    
            
    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []   
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self            
    
        
    def partial_fit(self, X, y):
        # done for the new input data
        # fit the data without reinitializing the weights
        
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        
        return self
        
        
    def _shuffle(self, X, y):
         r = np.random.permutation(len(y))       
         return X[r], y[r]
    
    
    def _initialize_weights(self, m):
        self.w_ = np.zeros(1+m)
        self.w_initialized = True
        
        
    def _update_weights(self, xi, target):
        # same as in previous adaline
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi * error
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
        
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    

    def activation(self, X):
        return self.net_input(X)
        
    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from numpy.random import seed
import pandas as pd
import numpy as np

# load data same as in perceptron model        
iris = load_iris()
X = pd.DataFrame(iris.data)
X = X.iloc[0:100, [0, 2]].values
y = pd.DataFrame(iris.target)
y = y.iloc[0:100].values
y = np.where(y==0, -1, 1)

from Decision_Boundary import plot_decision_regions

# feature scaling
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)

# for decision boundary
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()

# for convergence plot
plt.plot(range(1, len(ada.cost_)+1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average cost')
plt.show()