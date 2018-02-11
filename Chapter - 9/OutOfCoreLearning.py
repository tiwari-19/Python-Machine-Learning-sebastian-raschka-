#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 10:57:35 2017

@author: ashish
"""

# online algorithm for out-of-core learning
# using Stochastic gradient descent

import numpy as np
import re
from nltk.corpus import stopwords

stop = stopwords.words('english')

def tokenizer(text):
       text = re.sub('<[^>]*>', '', text)
       emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
       text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
       tokenized = [w for w in text.split() if w not in stop]
       return tokenized


def stream_docs(path):
       with open(path, 'r') as csv:
              next(csv)
              for line in csv:
                     text, label = line[:-3], (line[-2])
                     yield text, label
                     

def get_minibatch(doc_stream, size):
       docs, y = [], []
       try:
              for _ in range(size):
                     text, label = next(doc_stream)
                     docs.append(text)
                     y.append(label)
       except StopIteration:
              return None, None
       
       return docs, y
       