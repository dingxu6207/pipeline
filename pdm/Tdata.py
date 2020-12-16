# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 22:23:51 2020

@author: dingxu
"""

import numpy as np

data = np.loadtxt('prasemagv19.txt')

data = data.T

np.savetxt('prasemagv19.txt', data)