#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 13:01:44 2017

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


# --------------- Feature Scaling ------------------#

from Decision_Boundary import plot_decision_regions

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

ada = Adaline(n_iter=15, eta=0.01)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()