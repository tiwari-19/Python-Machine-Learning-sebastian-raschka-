#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 23:44:58 2017

@author: ashish
"""

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd


df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", header=None)
X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=1)
