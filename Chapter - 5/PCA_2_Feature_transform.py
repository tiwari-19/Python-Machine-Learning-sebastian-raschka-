#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 18:56:27 2017

@author: ashish
"""

from PCA_1_Cov_mat import eigen_vals, eigen_vecs, X_train_std
from Partitioning_Dataset import y_train
import matplotlib.pyplot as plt
import numpy as np

eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)

W = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print 'Matrix W:\n', W

X_train_pca = X_train_std.dot(W)

colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==l, 0],
                X_train_pca[y_train==l, 1],
                c=c, label=l, marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()