# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:05:25 2020

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt

datamag = np.loadtxt('datamag26.txt')

time = datamag[:,0]
mag = datamag[:,1]

plt.plot(time, mag, '.')
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向