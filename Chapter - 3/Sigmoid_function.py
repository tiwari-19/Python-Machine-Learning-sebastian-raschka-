#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 22:52:09 2017

@author: ashish
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return (1.0 / (1.0 + np.exp(-z)))
    
z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)

plt.plot(z, phi_z)
plt.yticks([0.0, 0.5, 1.0])
plt.ylim(-0.1, 1.1)
plt.axhline(y=0.5, ls='dotted')
plt.axvline(x=0.0, ls='solid', color='k')
plt.show()