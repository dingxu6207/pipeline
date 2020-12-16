# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 22:30:54 2020

@author: dingxu
"""

import matplotlib.pylab as plt
import numpy as np
from PyAstronomy.pyasl import foldAt
from PyAstronomy.pyTiming import pyPDM

path = 'E:\\shunbianyuan\\data\\lixuzhi\\datamag\\'
file = 'datamagv23.txt'

data = np.loadtxt(path+file)
data = data.T


time = data[:,0]
flux = data[:,1]

plt.figure(0)
plt.plot(time, flux, '.')
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向


phases = foldAt(time, 2.5)
sortIndi = np.argsort(phases)
# ... and, second, rearrange the arrays.
phases = phases[sortIndi]
flux = flux[sortIndi]


plt.figure(1)
plt.plot(phases, flux, '.')
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
plt.xlabel('Phrase',fontsize=14)
plt.ylabel('mag',fontsize=14)

prasemag = np.vstack((phases, flux))
np.savetxt('prasemagv23.txt', prasemag)

'''
S = pyPDM.Scanner(minVal=2.5, maxVal=3.25, dVal=0.01, mode="frequency")
P = pyPDM.PyPDM(time, flux)

#f1, t1 = P.pdmEquiBinCover(10, 3, S)
f2, t2 = P.pdmEquiBin(10, S)
plt.figure(2)
plt.plot(f2, t2, 'gp-')
#plt.plot(f1, t1, 'rp-')
plt.xlabel('freguency',fontsize=14)
plt.ylabel('delta',fontsize=14)

prasemag = np.vstack((f2, t2))
np.savetxt('PDMV10.txt', prasemag)
'''