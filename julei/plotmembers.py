# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 21:44:21 2020

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pandas as pd

data = np.loadtxt('BPRPG.txt')
data = data.T

plt.figure(0)
plt.xlim((-1,4))
plt.ylim((10,22))
plt.scatter(data[:,1], data[:,0], marker='o', color='lightcoral',s=10)
x_major_locator = MultipleLocator(1)
plt.xlabel('BP-RP',fontsize=14)
#plt.xlabel('G-RP',fontsize=14)
plt.ylabel('Gmag',fontsize=14)


bemeberdata = np.loadtxt('Be99member.txt')
hang = len(bemeberdata)
for i in range(10):
    G = bemeberdata[i,6]
    BPRP = bemeberdata[i,7]-bemeberdata[i,8]
    plt.scatter(BPRP, G, marker='o', color='darkgreen', s=20)
    plt.text(BPRP, G, 'V'+str(i+17), fontsize=10, color = "b", style = "italic", weight = "light", verticalalignment='center', horizontalalignment='left',rotation=0) #给散点加标签
        
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向

plt.figure(1)
pmdata = np.loadtxt('PM.txt')
pmdata = pmdata.T
plt.scatter(pmdata[:,1], pmdata[:,0], marker='o', color='lightcoral',s=10)
