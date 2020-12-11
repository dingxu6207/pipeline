# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 16:03:33 2020

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt('savedata.txt')
hang,lie = data.shape

data1 = data[1000,:]

plt.plot(data1[0:100], '.')
print(data1[100:104])


