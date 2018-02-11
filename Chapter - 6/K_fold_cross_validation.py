#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 19:03:18 2017

@author: ashish
"""

from breast_cancer_data import X_train, y_train
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import numpy as np


pipe_lr = Pipeline([('sc',StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('lr', LogisticRegression(random_state=1))])

kfold = StratifiedKFold(y=y_train, n_folds=10, random_state=1)
scores = []

from sklearn.cross_validation import cross_val_score
scores = cross_val_score(estimator=pipe_lr,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=-1)

print 'Cross validation scores: ', scores
print '\nCross Validation Accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))