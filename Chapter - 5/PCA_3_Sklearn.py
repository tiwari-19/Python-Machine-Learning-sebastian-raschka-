#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 10:24:23 2017

@author: ashish
"""

from Partitioning_Dataset import X_train, X_test, y_train, y_test
from Decision_Boundary import plot_decision_regions
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

    
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# PCA with two principal components
pca = PCA(n_components=2)
lr = LogisticRegression()

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

lr.fit(X_train_pca, y_train)

plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(loc='lower left')
plt.title('PCA using logistic regression')
plt.show()