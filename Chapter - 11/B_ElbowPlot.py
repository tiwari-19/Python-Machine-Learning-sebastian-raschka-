#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 20:59:58 2017

@author: ashish
"""
# Elbow method
# limitation of KMeans is that we have to assign K as a priori
# select the value of k at which the elbow is formed, here k=3

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

distortions = []

for i in range(1, 11):
       km = KMeans(n_clusters=i, 
                   init='k-means++',
                   n_init=10,
                   max_iter=300,
                   random_state=0)
       km.fit(X)
       distortions.append(km.inertia_)
       
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of cluster')
plt.ylabel('Distortion')
plt.show()