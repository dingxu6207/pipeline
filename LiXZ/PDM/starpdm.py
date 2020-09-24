# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 17:05:01 2020

@author: dingxu
"""

import matplotlib.pylab as plt
import numpy as np
from PyAstronomy.pyasl import foldAt
from PyAstronomy.pyTiming import pyPDM


#path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\pipeline\\LiXZ\\kongphometry\\'
#path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\pipeline\\LiXZ\\psfphpmetry\\'
pathfile = 'E:\\shunbianyuan\\phometry\\pipelinecode\\pipeline\\LiXZ\\star\\'+'datamag.txt'
lighttime = np.loadtxt(pathfile)
light = lighttime[1,:]
time = lighttime[0,:]

hang = 0
flux = light


plt.figure(0)
plt.plot(time, flux, '.')
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向


phases = foldAt(time, 0.6025294184988582)
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

'''
S = pyPDM.Scanner(minVal=1, maxVal=2, dVal=0.01, mode="frequency")
P = pyPDM.PyPDM(time, flux)

f1, t1 = P.pdmEquiBinCover(10, 3, S)
f2, t2 = P.pdmEquiBin(10, S)
plt.figure(2)
plt.plot(f2, t2, 'gp-')
plt.plot(f1, t1, 'rp-')
'''