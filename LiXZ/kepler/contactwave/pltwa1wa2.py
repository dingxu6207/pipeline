# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 21:10:20 2021

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt

data1 = np.loadtxt('mag1.txt')
data2 = np.loadtxt('mag2.txt')

hang1 = data1[0,:]
mag1 = data1[1,:]

hang2 = data2[0,:]
mag2 = data2[1,:]

plt.scatter(hang1, mag1, s=20, c='r', alpha=0.4)
plt.scatter(hang2, mag2, s=20, c='b', alpha=0.4)

ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向


plt.legend(('Johnson:V', 'Johnson:B'), loc='upper right')