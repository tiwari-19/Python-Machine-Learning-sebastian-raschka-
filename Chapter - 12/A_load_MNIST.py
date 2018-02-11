#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 18:05:52 2017

@author: ashish
"""

import os
import struct
import numpy as np

def load_mnist(path, kind='train'):
       labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
       images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
       
       with open(labels_path, 'rb') as lpath:
              magic, n = struct.unpack('>II', lpath.read(8))
              labels = np.fromfile(lpath, dtype=np.uint8)
       
       with open(images_path, 'rb') as imgpath:
              magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
              images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
       
       return images, labels

'''

X_train, y_train = load_mnist('MNIST_data', kind='train')
#X_test, y_test = load_mnist('MNIST_data', kind='test')

import matplotlib.pyplot as plt

print 'Sample of all digits'
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, )
ax = ax.flatten()
for i in range(10):
       img = X_train[y_train==i][0].reshape(28, 28)
       ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()


print '\n\nDifferent samples of same digit'
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True, )
ax = ax.flatten()

for i in range(25):
       img = X_train[y_train==5][i].reshape(28, 28)
       ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()


'''