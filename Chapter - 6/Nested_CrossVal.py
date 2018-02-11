#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 13:32:38 2017

@author: ashish
"""

# this is an example of 5 x 2 nested loop
# there are 5 outer folds and 2 inner folds

from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from breast_cancer_data import X_train, y_train
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import numpy as np

pipe_svc = Pipeline([('sc', StandardScaler()),
                    ('clf', SVC(random_state=1))])
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear']},
              {'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel':['rbf']}]

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=2,
                  n_jobs=-1)

scores = cross_val_score(estimator=gs,
                        X=X_train,
                        y=y_train,
                        scoring='accuracy',
                        cv=5)
print 'CV accuracy of SVM model: %0.3f +/- %0.3f' % (np.mean(scores), np.std(scores))

from sklearn.tree import DecisionTreeClassifier

param_grid = [{'max_depth':[1,2,3,4,5,6,7,None]}]

gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=5)
scores = cross_val_score(estimator=gs, 
                         X=X_train,
                         y=y_train,
                         scoring='accuracy',
                         cv=2)
print 'CV accuracy of DecisionTree model : %0.3f +/- %0.3f' % (np.mean(scores), np.std(scores))