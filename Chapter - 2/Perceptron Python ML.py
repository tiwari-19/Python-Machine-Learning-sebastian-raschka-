#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 11:38:11 2017

@author: ashish
"""

class Perecptron(object):
    
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):
        self.w_ = np.zeros( 1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update*xi
                self.w_[0] += update
                errors += int(update!=0.0)
            self.errors_.append(errors)
        return self
        
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
        
        
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

# load data same as in perceptron model        
iris = load_iris()
X = pd.DataFrame(iris.data)
# selecting only the first and third columns as features
X = X.iloc[0:100, [0, 2]].values

y = pd.DataFrame(iris.target)
# binary classification with classes -1 and 1
y = y.iloc[0:100].values
y = np.where(y==0, -1, 1)


plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.title("Actual distribution of data")
plt.show()

ppn = Perecptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('No.of Misclassifications')
plt.title('Convergence in the n\'th iteration')
plt.show()


# --------------Plotting the decision boundary -----------------#

from Decision_Boundary import plot_decision_regions

plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.title('Plot showing the Decision Boundary')
plt.show()