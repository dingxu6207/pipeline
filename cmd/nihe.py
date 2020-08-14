# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 21:47:28 2020

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt('BVce.txt')

Bce = data[:,0]
Vce = data[:,1]

plt.figure(0)
plt.plot(Bce-Vce,Vce,'.')
plt.xlabel('B-V')
plt.ylabel('V')
#ax = plt.gca()
#ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
#ax.invert_yaxis() #y轴反向


BV = np.loadtxt('BV.txt')
B = BV[:,0]
V = BV[:,1]
bxDATA = B-V+0.39
byDATA = V+12.36
#plt.figure(1)
plt.plot(bxDATA,byDATA)

plt.xlim((0.5,2))
plt.ylim((12,19))


ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向