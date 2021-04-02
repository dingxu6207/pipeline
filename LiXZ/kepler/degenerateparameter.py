# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 17:11:31 2021

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
#import seaborn as sns
#sns.set()

lightdata = np.loadtxt('savedatasample3.txt') 
#lightdata = np.loadtxt('alldatasample35.txt') 

lightdata = lightdata[lightdata[:,100] < 45]

for i in range(len(lightdata)):
    lightdata[i,0:100] = -2.5*np.log10(lightdata[i,0:100])
    lightdata[i,0:100] = lightdata[i,0:100] - np.mean(lightdata[i,0:100])
    

def calculater(ydata, caldata):
    res_ydata  = np.array(ydata) - np.array(caldata)
    ss_res     = np.sum(res_ydata**2)
    ss_tot     = np.sum((ydata - np.mean(ydata))**2)
    r_squared  = 1 - (ss_res / ss_tot)
    return r_squared

temp = []
staind = 200
onelightcurve = lightdata[staind,0:100]
print(lightdata[staind,100:104])

for i in range(len(lightdata)):
    R2 = calculater(onelightcurve, lightdata[i,0:100])
    #print(R2)
    temp.append(R2)
    
npR2 = np.array(temp)

tempi = []
for i in range(len(npR2)):
    if npR2[i]>0.9995:
        print(npR2[i])
        tempi.append(i)
       
xzuobiao = np.arange(0,1,0.01)
plt.figure(1)
#plt.plot(xzuobiao, lightdata[staind,0:100],'.')
plt.scatter(xzuobiao, lightdata[staind,0:100],s=40, c='b', alpha=0.4)
for i in range(len(tempi)):
    index = tempi[i]
    if lightdata[index,100]>43.4 or lightdata[index,100]<39:
        print(index)
        plt.plot(xzuobiao, lightdata[index,0:100],alpha = 0.7)
        print(lightdata[index,100:104])
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
plt.xlabel('phase',fontsize=14)
plt.ylabel('mag',fontsize=14)

plt.figure(2)
#plt.scatter(xzuobiao, lightdata[staind,0:100],s=40, c='none', alpha=0.9, edgecolors='r')
#plt.scatter(xzuobiao, lightdata[51995,0:100],marker='*',s=30, c='g', alpha=0.9)
#A, = plt.scatter(xzuobiao, lightdata[66343,0:100],s=40, c='none', alpha=0.9, edgecolors='r')
A, = plt.plot(xzuobiao, lightdata[staind,0:100],'^',alpha = 0.4,c='r')
B, = plt.plot(xzuobiao, lightdata[51995,0:100],'+',alpha = 1,c='g')
C, = plt.plot(xzuobiao, lightdata[66343,0:100],alpha = 0.9, linewidth=1, c='b')
ax1 = plt.gca()
ax1.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax1.invert_yaxis() #y轴反向
plt.xlabel('phase',fontsize=14)
plt.ylabel('mag',fontsize=14)
#设置主刻度标签的位置,标签文本的格式
#plt.xaxis.set_major_locator(xmajorLocator)
legend=plt.legend(handles=[A,B,C],labels=['target1','target2','target3'])
