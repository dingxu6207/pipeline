# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 23:56:01 2020

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

data = np.loadtxt('parallexg.txt')
data = data.T

BRG = np.loadtxt('pallexGDTA.txt')


highdataBPRP = data[:,0]
highdataGmag = data[:,1]

#plt.xlim((-1,4))
plt.ylim((-6,8))
plt.scatter(highdataBPRP, highdataGmag, marker='o', color='lightcoral',s=5)

for i in range(len(BRG)):
    plt.plot((BRG[i][1]), BRG[i][0], '.')
    plt.text((BRG[i][1]), BRG[i][0], str(1+i), fontsize=10, color = "b", style = "italic", weight = "light", verticalalignment='top', horizontalalignment='left',rotation=0)
    


