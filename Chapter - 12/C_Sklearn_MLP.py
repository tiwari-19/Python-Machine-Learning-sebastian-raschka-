#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 10:41:16 2017

@author: ashish
"""

from A_load_MNIST import load_mnist
from sklearn.neural_network import MLPClassifier

X_train, y_train = load_mnist('MNIST_data', kind='train')
nn = MLPClassifier(hidden_layer_sizes=(50,),
                   random_state=1,
                   batch_size=50)

nn.fit(X_train, y_train)

print 'Train accuracy %0.2f' % (nn.score(X_train, y_train) *100)

X_test, y_test = load_mnist('MNIST_data', kind='t10k')
print 'Test accuracy: %0.2f' % (nn.score(X_test, y_test) * 100)
















