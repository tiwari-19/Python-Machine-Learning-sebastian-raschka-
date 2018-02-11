#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 21:21:03 2017

@author: ashish
"""

from sklearn.cross_validation import train_test_split
from Decision_Boundary import plot_decision_regions
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np


iris = load_iris()
X = iris.data[:, [2,3]]
y = iris.target

# split data into train and test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# for standard feature scaling
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

# load perceptron model
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(x_train_std, y_train)

y_pred = ppn.predict(x_test_std)
acc = accuracy_score(y_test, y_pred)
print 'Accuracy score: %.2f' %acc

# combine the splitted data
x_combine = np.vstack((x_train_std, x_test_std))
y_combine = np.hstack((y_train, y_test))
plot_decision_regions(x_combine, y_combine, ppn, test_idx=range(105, 150))
plt.xlabel('Petal length [cm]')
plt.ylabel('Petal width [cm]')
plt.legend(loc='upper left')
plt.show()