# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 13:37:14 2020

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

lightdata = np.loadtxt('savedatasample2.txt') 

import seaborn as sns
sns.set()


lightdata = lightdata[lightdata[:,100]>70]
lightdata = lightdata[lightdata[:,101]<0.3]
lightdata = lightdata[lightdata[:,103]<1.04]
lightdata = lightdata[lightdata[:,103]>0.92]

plt.figure(0)
incldata = lightdata[:,100]
sns.kdeplot(incldata,shade=True)

plt.figure(1)
qdata = lightdata[:,101]
sns.kdeplot(qdata,shade=True)

plt.figure(2)
rdata = lightdata[:,102]
sns.kdeplot(rdata,shade=True)

plt.figure(3)
tdata = lightdata[:,103]
sns.kdeplot(tdata,shade=True)

plt.figure(5)
ax3d = plt.gca(projection="3d")
ax3d.scatter3D(lightdata[:,100], lightdata[:,101], lightdata[:,102], c ='b', marker='o', s=10)


tdata = np.copy(lightdata)
tdata = tdata[tdata[:,103]>0.999]
tdata = tdata[tdata[:,103]<1.001]

plt.figure(4)
ax3d = plt.gca(projection="3d")
ax3d.scatter3D(tdata[:,100], tdata[:,101], tdata[:,102], c ='b', marker='o', s=10)


