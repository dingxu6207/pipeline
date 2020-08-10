# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 15:51:18 2020

@author: dingxu
"""
import numpy as np
import matplotlib.pyplot as plt

file = 'E:\\shunbianyuan\\phometry\\'+'NGC7142.txt'
data = np.loadtxt(file)

Gaia_BP = data[:,1]
Gaia_RP = data[:,2]

xdata = Gaia_BP-Gaia_RP
ydata = data[:,0]

plt.figure(0)
plt.plot(xdata,ydata)
plt.xlabel('G(BP-RP)')
plt.ylabel('G')
ax = plt.gca()
ax.yaxis.set_ticks_position('right') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向


BV = np.loadtxt('data.txt')
B = BV[:,0]
V = BV[:,1]
bxDATA = B-V
byDATA = V
plt.figure(1)
plt.plot(bxDATA,byDATA)
ax = plt.gca()
ax.yaxis.set_ticks_position('right') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向