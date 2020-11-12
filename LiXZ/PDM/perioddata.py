# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 22:30:54 2020

@author: dingxu
"""

import matplotlib.pylab as plt
import numpy as np
from PyAstronomy.pyasl import foldAt
from PyAstronomy.pyTiming import pyPDM
from matplotlib.pyplot import MultipleLocator

path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\pipeline\\LiXZ\\kongphometry\\'
#path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\pipeline\\LiXZ\\psfphpmetry\\'
light = np.loadtxt(path+'arrayjiaocha.txt')
time = np.loadtxt(path+'datatime.txt')
hang = 59
flux = light[hang,2:]


plt.figure(0)
plt.plot(time, flux, '.')
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向

'''
phases = foldAt(time, 0.3636)
sortIndi = np.argsort(phases)
# ... and, second, rearrange the arrays.
phases = phases[sortIndi]
flux = flux[sortIndi]


plt.figure(1)
plt.plot(phases, flux, '.')
#y_major_locator=MultipleLocator(0.5)

ax = plt.gca()
#ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
plt.xlabel('Phrase',fontsize=14)
plt.ylabel('mag',fontsize=14)
#plt.ylim(7.0,3.0)

'''
S = pyPDM.Scanner(minVal=2.3, maxVal=3.0, dVal=0.01, mode="frequency")
P = pyPDM.PyPDM(time, flux)

#f1, t1 = P.pdmEquiBinCover(10, 3, S)
f2, t2 = P.pdmEquiBin(10, S)
plt.figure(2)
plt.plot(f2, t2, 'gp-')
#plt.plot(f1, t1, 'rp-')
plt.xlabel('frequency',fontsize=14)
plt.ylabel('Theta', fontsize=14)
