# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 01:03:33 2020

@author: dingxu
"""

import matplotlib.pyplot as plt
import numpy as np

lighttime = np.loadtxt('datamag.txt')
light = lighttime[1,:]
time = lighttime[0,:]

cha = np.loadtxt('jiaoyan.txt')


plt.figure(figsize=(7,5))
plt.subplot(211)  #两行一列,第一个图
plt.plot(time, light, '.')
plt.ylabel('V-C')
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向


plt.subplot(212) #两行一列.第二个图
plt.plot(time, cha, '.')
plt.xlabel('JD')
plt.ylabel('C-CH')
