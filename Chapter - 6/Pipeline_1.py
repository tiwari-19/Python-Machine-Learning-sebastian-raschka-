#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 00:01:49 2017

@author: ashish
"""

from breast_cancer_data import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


pipe_lr = Pipeline([('sc1',StandardScaler()),
                    ('pca1', PCA(n_components=2)),
                    ('lr1', LogisticRegression(random_state=1))])
pipe_lr.fit(X_train, y_train)
print "Test Accuracy = %0.3f" % pipe_lr.score(X_test, y_test)