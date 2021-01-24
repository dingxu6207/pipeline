# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 00:37:35 2020

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import interpolate

'''
phraseflux = np.loadtxt('D:\\Phoebe\\data\\v737per.B')

phrase = phraseflux[:,0]

flux = phraseflux[:,1]

sortIndi = np.argsort(phrase)
phrase = phrase[sortIndi]
flux = flux[sortIndi]

ar8 = np.vstack((phrase,flux))
np.savetxt('V737.txt', ar8.T)
'''

#data = np.loadtxt('V737.txt') #CUTau_Qian2005B.nrm
data = np.loadtxt('UVLyn_Vanko2001B.nrm')
phrase = data[:,0]
flux = data[:,1]

plt.figure(0)
plt.plot(phrase,flux,'.')#对原始数据画散点图

ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
plt.xlabel('Phase',fontsize=14)
plt.ylabel('mag',fontsize=14)


sx1 = np.linspace(0,1,500)
func1 = interpolate.UnivariateSpline(phrase, flux,s=0.0055)#强制通过所有点
sy1 = func1(sx1)

plt.plot(sx1, sy1,'.', c='r')#对原始数据画散点图
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向

interdata = np.vstack((sx1,sy1))
np.savetxt('V737inter.txt', interdata.T)
