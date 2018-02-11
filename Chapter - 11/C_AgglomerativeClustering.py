#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 21:16:05 2017

@author: ashish
"""

from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score

X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

ac = AgglomerativeClustering(n_clusters=3,
                             affinity='euclidean', 
                             linkage='complete')

labels = ac.fit_predict(X)
print labels
