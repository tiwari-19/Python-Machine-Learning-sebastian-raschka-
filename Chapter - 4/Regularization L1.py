#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 17:06:18 2017

@author: ashish
"""

from Partitioning_Dataset import X_train, X_test, y_train, y_test
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

lr = LogisticRegression(penalty='l1', C=0.1)
lr.fit(X_train_std, y_train)
print "\nTraining Accuracy:", lr.score(X_train_std, y_train)
print "Test Accuracy:", lr.score(X_test_std, y_test)

print "\nThe intercepts are:\n", lr.intercept_
print "\nThe weight matrix is:\n", lr.coef_

