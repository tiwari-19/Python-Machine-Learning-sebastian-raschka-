#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 05:38:00 2017

@author: ashish
"""

import pandas as pd
from io import StringIO

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

csv_data = unicode(csv_data)
df = pd.read_csv(StringIO(csv_data))

print "Original data:\n", df

print "\nNumber of missing values in each column:"
print df.isnull().sum()

# Methods to handle missing values

# Method 1: Remove the entire row/col containing the missing values, mostly
# not feasible as we may lose valuable data
print "\nRemoving all rows containing NaN\n", df.dropna()
print "\nRemoving all columns containing NaN\n", df.dropna(axis=1)
print "\nRemoving rows where NaN appears in particular col:\n", df.dropna(subset=['C'])


# Method 2: Interpolation techniques: techniques for dealing with missing values
# Most common interpolation techniques: Mean Imputation
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
# changing axis=1 would calculate row means
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
print "\nData after Mean Imputation\n", imputed_data