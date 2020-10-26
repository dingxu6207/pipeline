# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 22:20:43 2020

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('dataxie1.txt')
mr1data = np.loadtxt('MR1.txt')
mr2data = np.loadtxt('MR2.txt')


M1 = data[:,0]
R1 = data[:,1]

M2 = data[:,2]
R2 = data[:,3]

mr1hang = mr1data[:,0]
mr1lie = mr1data[:,1]

mr2hang = mr2data[:,0]
mr2lie = mr2data[:,1]


plt.plot(M1,R1,'*')
plt.plot(M2,R2,'.')

plt.plot(mr1hang, mr1lie, linewidth=5)
plt.plot(mr2hang, mr2lie, linewidth=3)

hang,lie = data.shape

for i in range (0,hang):
    
    x10 = data[i,0]
    y10 = data[i,1]
    
    x11 = data[i,2]
    y11 = data[i,3]
    plt.plot([x10,x11],[y10,y11],linewidth = 0.8)
    
plt.xlabel('logM',fontsize=14)
plt.ylabel('logR',fontsize=14)