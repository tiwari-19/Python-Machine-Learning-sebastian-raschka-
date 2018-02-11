#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 10:26:53 2017

@author: ashish
"""

from breast_cancer_data import X_train, y_train, X_test, y_test
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

pipe_svc = Pipeline([('sc', StandardScaler()),
                     ('clf', SVC(random_state=1))])

param_grid = [{'clf__C': param_range, 'clf__kernel':['linear']},
              {'clf__C': param_range, 'clf__kernel':['rbf'], 'clf__gamma':param_range}]

gs = GridSearchCV(estimator=pipe_svc, 
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(X_train, y_train)
print gs.best_params_
print gs.best_score_

clf = gs.best_estimator_
clf.fit(X_train, y_train)
print 'Test Accuracy: %0.3f' % clf.score(X_test, y_test)

import matplotlib.pyplot as plt
from Decision_Boundary import plot_decision_regions

plot_decision_regions(X_train, y_train, clf)
plt.show()