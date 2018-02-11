#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 06:13:51 2017

@author: ashish
"""

'''
Handling Categorical data
Categorical data is of two types: 
    Ordinal (that can be sorted or ordered, eg. T-shirt size XL>L>M)
    Nominal (that cannot be compared or ordered, eg. T-shirt colour)
'''

import pandas as pd
df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                   ['red', 'L', 13.5, 'class2'],
                   ['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']

# color - nominal
# price - numerical
# size - ordinal
print df

# Mapping ordinal features
size_mapping = {'XL':3, 'L':2, 'M':1}
df['size'] = df['size'].map(size_mapping)
print "\nData after size mapping:\n", df

# similarly for classlabels
# there is also a LabelEncoder function in sklearn.preprocessing for this
import numpy as np
class_mapping = {label:idx for idx,label in enumerate(np.unique(df['classlabel']))}
df['classlabel'] = df['classlabel'].map(class_mapping)
print "\nData after classlabel mapping:\n", df


# for mapping nominal feature we use One hot encoding
# eg. blue color will be blue=1, red=0, green=0
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:,0] = color_le.fit_transform(X[:, 0])
ohe = OneHotEncoder(categorical_features=[0])
# categorial_features takes the column number
print "\nAfter color mapping using onehotencoder:\n", ohe.fit_transform(X).toarray()

# more efficient way
df = pd.get_dummies(df[['price', 'color', 'size']])
print "\nOne hot encoding using pandas feature get_dummies:\n",df