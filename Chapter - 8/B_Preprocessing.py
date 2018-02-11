#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 10:20:47 2017

@author: ashish
"""

import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

porter = PorterStemmer()

# removing HTML tags and punctuations but not emoticons
def preprocessor(text):
       text = re.sub('<[^>]*>', '', text)
       emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
       
       #adding emoticons at the end
       text = re.sub('[\W]+', ' ', text.lower()) + ''.join(emoticons).replace('-', '')
       return text


def tokenizer(text):
       return text.split()


# PorterStemmer reduces the word to its root form ex. running -> run
def tokenizer_porter(text):
       return [porter.stem(word) for word in text.split()]


stop = stopwords.words('english')