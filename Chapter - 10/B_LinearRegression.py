#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 17:40:58 2017

@author: ashish
"""

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('housing_data.csv')
X = df[['RM']].values
y = df['MEDV'].values

slr = LinearRegression()
slr.fit(X, y)
print 'Slope of the graph %0.3f' % slr.coef_[0]
print 'Intercept: %0.3f' % slr.intercept_



plt.scatter(X, y, c='blue')
plt.plot(X, slr.predict(X), c='red')
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.show()
