#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 23:25:12 2017

@author: ashish
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('housing_data.csv')
X = df[['LSTAT']].values
y = df['MEDV'].values

# create polynomial features
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)


# linear fit
X_lin = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
regr = LinearRegression()
regr = regr.fit(X, y)
y_lin = regr.predict(X_lin)
r2_lin = r2_score(y, regr.predict(X))


# quadratic fit
X_quad = quadratic.fit_transform(X)
regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_lin))
r2_quad = r2_score(y, regr.predict(X_quad))


# cubic fit
X_cubic = cubic.fit_transform(X)
regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_lin))
r2_cubic = r2_score(y, regr.predict(X_cubic))


# plotting all
plt.scatter(X, y, label='Training Points', color='lightgrey')
plt.plot(X_lin, y_lin, label='linear (d=1), $R^2=%.2f$' % r2_lin, color='blue', lw=2, linestyle=':')
plt.plot(X_lin, y_quad_fit, label='quadratic (d=2), $R^2=%.2f$' % r2_quad, color='red', lw=2, linestyle='-')
plt.plot(X_lin, y_cubic_fit, label='cubic (d=3), $R^2=%.2f$' % r2_cubic, color='green', lw=2, linestyle='--')
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper right')
plt.show()