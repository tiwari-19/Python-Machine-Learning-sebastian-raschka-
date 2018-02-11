#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 22:12:20 2017

@author: ashish
"""

import pyprind
import pandas as pd
import numpy as np
import os

pbar = pyprind.ProgBar(50000)
labels = {'pos':1, 'neg':0}
df = pd.DataFrame()

for s in ('test', 'train'):
       for l in ('pos', 'neg'):
              path ='./aclImdb/%s/%s' % (s, l)
              for file in os.listdir(path):
                     with open(os.path.join(path, file), 'r') as infile:
                            txt = infile.read()
                            df = df.append([[txt, labels[l]]], ignore_index=True)
                            pbar.update()
df.columns = ['review', 'sentiment']
#print 'Dataframe created !!'

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('./movie_data.csv', index=False)
#df = pd.read_csv('./movie_data.csv')
#print 'Sample data:'
#print df.head(3)