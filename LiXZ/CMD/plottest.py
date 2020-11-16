# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 23:02:05 2020

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

data = np.loadtxt('BPRPG.txt')
data = data.T

BRG = np.loadtxt('BPRPGDATA.txt')


highdataBPRP = data[:,1]
highdataGmag = data[:,0]

plt.xlim((-1,4))
plt.ylim((10,22))
plt.scatter(highdataBPRP, highdataGmag, marker='o', color='lightcoral',s=5)

for i in range(len(BRG)):
    plt.plot((BRG[i][0]-BRG[i][1]), BRG[i][2], '.')
    plt.text((BRG[i][0]-BRG[i][1]), BRG[i][2], str(1+i), fontsize=10, color = "b", style = "italic", weight = "light", verticalalignment='baseline', horizontalalignment='left',rotation=0)
    
x_major_locator = MultipleLocator(1)


plt.xlabel('BP-RP',fontsize=14)
plt.ylabel('Gmag',fontsize=14)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
