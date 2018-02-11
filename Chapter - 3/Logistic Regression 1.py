#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 23:42:15 2017

@author: ashish
"""

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from Decision_Boundary import plot_decision_regions
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np


iris = load_iris()
X, y = iris.data[:, [2,3]], iris.target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(x_train_std, y_train)


x_combine = np.vstack((x_train_std, x_test_std))
y_combine = np.hstack((y_train, y_test))
plot_decision_regions(x_combine, y_combine, lr)

plt.xlabel('Petal length [cm]')
plt.ylabel('Petal width [cm]')
plt.legend(loc='upper left')
plt.show()