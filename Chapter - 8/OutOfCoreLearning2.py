#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 11:22:04 2017

@author: ashish
"""

from OutOfCoreLearning import *
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

vect = HashingVectorizer(decode_error='ignore', 
                         preprocessor=None, 
                         tokenizer=tokenizer, 
                         n_features=2**21)

clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
doc_stream = stream_docs(path='./movie_data.csv')

import numpy as np
import pyprind
pbar = pyprind.ProgBar(45)
classes = np.array(['0', '1'])


for _ in range(45):
       X_train, y_train = get_minibatch(doc_stream, size=1000)
       
       if not X_train:
              break
       X_train = vect.transform(X_train)
       clf = clf.partial_fit(X_train, y_train, classes=classes)
       pbar.update()
       
X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)

print('Accuracy: %.3f' % clf.score(X_test, y_test))

import pickle
import os
dest = os.path.join('movieclassifier', 'pkl_objects')
if not os.path.exists(dest):
       os.makedirs(dest)
pickle.dump(stop, open(os.path.join(dest, 'stopwords.pkl'),'wb'), protocol=2)
pickle.dump(clf, open(os.path.join(dest, 'classifier.pkl'), 'wb'),protocol=2)