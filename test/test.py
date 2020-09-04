# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 23:41:54 2020

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt

path = 'E:\\shunbianyuan\\phometry\\data\\'
file1 = 'KOMG.txt'
file2 = 'PSF.txt'
data1 = np.loadtxt(path+file1)
data2 = np.loadtxt(path+file2)

timedata = np.loadtxt(path+'datatime.txt')

plt.figure(0)
#plt.plot(timedata, (data2-data1), '.')

plt.plot(timedata, data1, '.', c='red', label = 'aperture')
plt.plot(timedata, data2, '.', c='blue', label = 'psf')
plt.legend(loc='best')
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向