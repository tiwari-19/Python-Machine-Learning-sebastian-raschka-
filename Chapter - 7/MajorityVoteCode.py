#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 22:24:49 2017

@author: ashish
"""

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


iris = load_iris()
# we are using only two features and two classes
X = iris.data[50:, [1,2]]
y = iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

# we will evaluate the performanc of three classifiers
clf1 = LogisticRegression(penalty='l2', C=0.001, random_state=0)
clf2 = DecisionTreeClassifier(criterion='entropy', max_depth=1, random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1, metric='minkowski', p=2)

pipe1 = Pipeline([('sc', StandardScaler()), ('clf', clf1)])
pipe3 = Pipeline([('sc', StandardScaler()), ('clf', clf3)])

clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN']
print '10-fold cross validation results without ensemble learning:'

for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
       scores = cross_val_score(estimator=clf,
                                X=X_train,
                                y=y_train,
                                cv=10,
                                scoring='roc_auc')
       print 'ROC AUC: %0.2f (+/- %0.2f) [%s]' %(scores.mean(), scores.std(),label)


# Now we combine these classifiers into ensemble classifier
from MajorityVoteClassifier import MajorityVoteClassifier
mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
scores = cross_val_score(estimator=mv_clf, 
                         X=X_train, 
                         y=y_train,
                         cv=10,
                         scoring='roc_auc')
print '\n10-fold cross validation result using majority voting:'
print 'ROC AUC: %0.2f (+/- %0.2f) [%s]' %(scores.mean(), scores.std(), 'Majority Voting')


# plotting Receiver Operator Characteristics Area Under the Curve ROC AUC
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

all_clfs = [pipe1, clf2, pipe3, mv_clf]
clf_labels += ['Majority Voting']
colors =['pink', 'red', 'green', 'blue']
linestyles = [':', '--', '-.', '-']

for clf, label, clr, style in zip(all_clfs, clf_labels, colors, linestyles):
       clf = clf.fit(X_train, y_train)
       y_pred = clf.predict_proba(X_test)[:, 1]
       fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
       roc_auc = auc(x=fpr, y=tpr)
       plt.plot(fpr, tpr, color=clr, linestyle=style, label='%s (auc = %0.2f)' % (label, roc_auc))
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()