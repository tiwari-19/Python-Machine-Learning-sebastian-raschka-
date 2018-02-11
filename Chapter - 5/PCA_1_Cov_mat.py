#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 18:28:28 2017

@author: ashish
"""

from Partitioning_Dataset import X_train, X_test
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print 'Eigen values of the co-variance matrix are:\n', eigen_vals

tot = sum(eigen_vals)

# variance explained ratio of an eigenvalues
var_exp = [i/tot for i in sorted(eigen_vals, reverse=True)]

# cummulative sum
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1,14),
        var_exp,
        alpha=0.5, align='center', 
        label='individual explained variance')
plt.step(range(1,14),
         cum_var_exp,
         where='mid',
         label='cummulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()