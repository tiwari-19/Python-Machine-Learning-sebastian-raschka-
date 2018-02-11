#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 19:16:34 2017

@author: ashish
"""

# RANdom SAmple Consensus (RANSAC) algorithm,
# which fits a regression model to a subset of the data, the so-called inliers.

from sklearn.linear_model import RANSACRegressor, LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('housing_data.csv')
X = df[['RM']].values
y = df['MEDV'].values

ransac = RANSACRegressor(base_estimator=LinearRegression(),
                         min_samples=50,
                         residual_threshold=5.0,
                         residual_metric=lambda x:np.sum(np.abs(x), axis=1),
                         max_trials=100,
                         random_state=0)
ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])

print '\nSlope of the graph: %0.3f' %(ransac.estimator_.coef_[0])
print 'Intercept: %.3f' %(ransac.estimator_.intercept_)

plt.scatter(X[inlier_mask], y[inlier_mask], c='blue', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], c='lightgreen', marker='s', label='Outliers')

plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel('Average number of rooms [RM]')
plt.xlabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper left')
plt.show()
