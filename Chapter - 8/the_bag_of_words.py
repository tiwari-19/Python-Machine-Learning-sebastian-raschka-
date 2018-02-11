#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 10:00:31 2017

@author: ashish
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(encoding='utf-8')
# for ngram, assign ngram_range = (n, n) in CountVectorizer

docs = np.array(['The sun is shining',
                 'The weather is sweet',
                 'The sun is shining and the weather is sweet'])
bag = count.fit_transform(docs)
print 'Vocabulary created as:', count.vocabulary_

print '\nFeature Vector for above:\n', bag.toarray()

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
np.set_printoptions(precision=2)

print '\ntf-idf for above feature vector'
print tfidf.fit_transform(bag).toarray()
