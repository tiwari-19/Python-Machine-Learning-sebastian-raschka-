#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 19:07:41 2017

@author: ashish
"""

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from OutOfCoreLearning import *
import numpy as np
import pickle
import os


import time
start_time = time.time()


# Logistic Classifier training 
path = os.getcwd()
path = os.chdir('../Chapter - 8')

vect = HashingVectorizer(decode_error='ignore', 
                         preprocessor=None, 
                         tokenizer=tokenizer, 
                         n_features=2**21)

clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
doc_stream = stream_docs(path='./movie_data.csv')

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


# Saving the current state of classifier using pickle       
os.chdir('../Chapter - 9')
dest = os.path.join('pkl_obj')
if not os.path.exists(dest):
       os.makedirs(dest)

pickle.dump(stop, open(os.path.join(dest, 'stopwords.pkl'), 'wb'), protocol=2)
pickle.dump(clf, open(os.path.join(dest, 'classifier.pkl'), 'wb'), protocol=2)
print 'Current state of classifier is saved as pickle object !!'

print("Time elapsed : %s seconds " % (time.time() - start_time))