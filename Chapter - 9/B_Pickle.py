#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 06:40:48 2017

@author: ashish
"""

# load the already saved state of classifier
import pickle
import os
import re

clf = pickle.load(open(os.path.join('pkl_obj', 'classifier.pkl'), 'rb'))


# now let's test it on some new reviews
import numpy as np
from vectorizer import vect

label = {'1':'positive', '0':'negative'}
example = ['I love this movie']
X = vect.transform(example)
print 'Prediction:', label[clf.predict(X)[0]]
print 'Prediction Probability:', np.max(clf.predict_proba(X))*100