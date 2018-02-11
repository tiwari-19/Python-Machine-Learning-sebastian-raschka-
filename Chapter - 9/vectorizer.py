#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 06:57:59 2017

@author: ashish
"""

import os
import re
import pickle
from sklearn.feature_extraction.text import HashingVectorizer

stop = pickle.load(open(os.path.join('pkl_obj', 'stopwords.pkl'), 'rb'))

def tokenizer(text):
       text = re.sub('<[^>]*>', '', text)
       emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
       text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
       tokenized = [w for w in text.split() if w not in stop]
       return tokenized

vect = HashingVectorizer(decode_error='ignore',
                         tokenizer=tokenizer, 
                         n_features=2**21)