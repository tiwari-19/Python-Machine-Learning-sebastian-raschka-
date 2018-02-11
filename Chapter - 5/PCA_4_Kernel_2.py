#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 20:56:40 2017

@author: ashish
"""

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

# radial basis function or gaussian function
def rbf_kernel_pca(X, gamma, n_components):
    sq_dists = pdist(X, 'sqeuclidean')
    mat_sq_dists = squareform(sq_dists)
    K = exp(-gamma * mat_sq_dists)
    
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K -one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    eigvals, eigvecs = eigh(K)
    
    X_pc = np.column_stack((eigvecs[:, -i] for i in range(1, n_components+1)))
    
    return X_pc


from sklearn.datasets import make_circles   
import matplotlib.pyplot as plt

X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
p1,p2 = ax[0], ax[1]

X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)

p1.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
p1.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)

p2.scatter(X_kpca[y==0, 0], np.zeros((500,1))+0.02, color='red', marker='^', alpha=0.5)
p2.scatter(X_kpca[y==1, 0], np.zeros((500,1))-0.02, color='blue', marker='o', alpha=0.5)

p1.set_xlabel('PC1')
p1.set_ylabel('PC2')
p2.set_ylim([-1, 1])
p2.set_yticks([])
p2.set_xlabel('PC1')

plt.show()  