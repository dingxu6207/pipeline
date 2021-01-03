# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 00:37:35 2020

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import interpolate
#phraseflux = np.loadtxt('mag.txt') #
phraseflux = np.loadtxt('D:\\Phoebe\\data\\v737per.B')

phrase = phraseflux[:,0]

flux = phraseflux[:,1]

sortIndi = np.argsort(phrase)
phrase = phrase[sortIndi]
flux = flux[sortIndi]



a=np.polyfit(phrase,flux,17)#用2次多项式拟合x，y数组
b=np.poly1d(a)#拟合完之后用这个函数来生成多项式对象
c=b(phrase)#生成多项式对象之后，就是获取x在这个多项式处的值
plt.figure(0)
plt.scatter(phrase,flux,marker='o',label='original datas')#对原始数据画散点图
plt.plot(phrase,c,ls='--',c='red',label='fitting with second-degree polynomial')#对拟合之后的数据，也就是x，c数组画图
plt.legend()

plt.figure(1)
plt.plot(phrase,c,'.')
phrasefluxdata = np.vstack((phrase, c))
np.savetxt('lightcurve.txt', phrasefluxdata.T)


