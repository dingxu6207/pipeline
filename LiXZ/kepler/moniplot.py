# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 22:27:20 2021

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import phoebe


yuandata1 = np.loadtxt('samle\\yuandata1.txt')
yuanmag1 = np.loadtxt('samle\\mag1.txt')

yuandata2 = np.loadtxt('samle\\yuandata2.txt')
yuanmag2 = np.loadtxt('samle\\mag2.txt')

yuandata3 = np.loadtxt('samle\\yuandata3.txt')
yuanmag3 = np.loadtxt('samle\\mag3.txt')

yuandata4 = np.loadtxt('samle\\yuandata4.txt')
yuanmag4 = np.loadtxt('samle\\mag4.txt')

datax = np.arange(0,100,1)/100

def fluxmag(dataflux):
    redata = -2.5*np.log10(dataflux)
    datamag = redata[0:100]-np.mean(redata[0:100])
    
    return datamag

datay1 = fluxmag(yuandata1)
datay2 = fluxmag(yuandata2)
datay3 = fluxmag(yuandata3)
datay4 = fluxmag(yuandata4)

plt.figure(0)
plt.plot(datax, datay1, '.', c='blue')
A = plt.scatter(datax, yuanmag1, c='none',marker='o',edgecolors='r', s=40) 

plt.plot(datax, datay2, '.', c='blue')
B = plt.scatter(datax, yuanmag2, c='none',marker='o',edgecolors='g', s=40)  

#plt.plot(datax, datay3, '.')
#C = plt.scatter(datax, yuanmag3, c='none',marker='o',edgecolors='b', s=40)  

#plt.plot(datax, datay4, '.', c='blue')
#D = plt.scatter(datax, yuanmag4, c='none',marker='o',edgecolors='y', s=40)     

#legend=plt.legend(handles=[A,B,C,D],labels=['Target1', 'Target2', 'Target3', 'Target4']) 
legend=plt.legend(handles=[A,B],labels=['incl=86.52,q=0.058,r=0.63,T2/T1=0.80', 'incl=89.48,q=0.97,r=0.48,T2/T1=0.89']) #86.5277 0.058286 0.63756 0.803279




ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
plt.xlabel('Phase',fontsize=14)
plt.ylabel('mag',fontsize=14)


plt.figure(1)

plt.plot(datax, datay3, '.', c='blue')
C = plt.scatter(datax, yuanmag3, c='none',marker='o',edgecolors='red', s=40)  

plt.plot(datax, datay4, '.', c='blue')
D = plt.scatter(datax, yuanmag4, c='none',marker='o',edgecolors='green', s=40)     

legend=plt.legend(handles=[C,D],labels=['incl=89.96,q=0.70,r=0.49,T2/T1=0.83', 'incl=50.59,q=0.81,r=0.50,T2/T1=0.98']) 


ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
plt.xlabel('Phase',fontsize=14)
plt.ylabel('mag',fontsize=14)