# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 22:30:54 2020

@author: dingxu
"""

import matplotlib.pylab as plt
import numpy as np
from PyAstronomy.pyasl import foldAt
from PyAstronomy.pyTiming import pyPDM

path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\pipeline\\star\\'
file = 'datamag.txt'

data = np.loadtxt(path+file)

time = data[0,:]
flux = data[1,:]


phases = foldAt(time, 0.5891016200294551)

sortIndi = np.argsort(phases)
# ... and, second, rearrange the arrays.
phases = phases[sortIndi]
flux = flux[sortIndi]


plt.figure(0)
plt.plot(phases, flux, '.')
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向


'''
S = pyPDM.Scanner(minVal=1, maxVal=2, dVal=0.1, mode="frequency")
P = pyPDM.PyPDM(time, flux)

f2, t2 = P.pdmEquiBin(10, S)
plt.figure(1)
plt.plot(f2, t2, 'gp-')
'''