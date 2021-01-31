# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:52:45 2020

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt

filetemp = 'E:\\shunbianyuan\\phometry\\pipelinecode\\pipeline\\LiXZ\\kepler\\data\\'


fileincl = 'incl.txt'

filedelT = 'divT.txt'

fileq = 'q.txt'

filer = 'r.txt'


dataincl = np.loadtxt(filetemp+fileincl)
#dataincl = dataincl.T

datadivt = np.loadtxt(filetemp+filedelT)

dataq = np.loadtxt(filetemp+fileq)

datar = np.loadtxt(filetemp+filer)

plt.figure(0)
sigma = np.std(dataincl[0,:]-dataincl[1,:])
sigma = round(sigma, 4)
plt.plot(dataincl[0,:], dataincl[0,:]-dataincl[1,:]+48, '.', c='blue')
plt.plot(dataincl[0,:], dataincl[1,:], '.', color='darkorange')
plt.plot(dataincl[0,:], dataincl[0,:], '-', c='black')
plt.axhline(y=48, color='r', linestyle='-')
plt.text(60, 50, 'y=48  '+'σ='+str(sigma), fontsize=14, color = "b", style = "italic")
#plt.title('incl', fontsize=20)
plt.xlabel('incl',fontsize=14)
plt.ylabel('predict-incl',fontsize=14)
plt.savefig('incl.jpg')

plt.figure(1)
sigma = np.std(datadivt[0,:]/100-datadivt[1,:]/100)
sigma = round(sigma, 4)
plt.plot(datadivt[0,:]/100, datadivt[1,:]/100, '.', color='darkorange')
plt.plot(datadivt[0,:]/100, datadivt[0,:]/100, '-', c='black')
plt.plot(datadivt[0,:]/100, (datadivt[0,:]-datadivt[1,:])/100+0.76, '.', c='blue')
plt.axhline(y=0.76, color='r', linestyle='-')
plt.text(0.85, 0.8, 'y=0.76  '+'σ='+str(sigma), fontsize=14, color = "b", style = "italic")
plt.xlabel('T2/T1',fontsize=14)
plt.ylabel('predict-T2/T1',fontsize=14)
plt.savefig('T1divT2.jpg')

plt.figure(2)
sigma = np.std(dataq[0,:]/100-dataq[1,:]/100)
sigma = round(sigma, 4)
plt.plot(dataq[0,:]/100, dataq[1,:]/100, '.', color='darkorange')
plt.plot(dataq[0,:]/100, dataq[0,:]/100, '-', c='black')
plt.plot(dataq[0,:]/100, (dataq[0,:]-dataq[1,:])/100, '.', c='blue')
plt.axhline(y=0, color='r', linestyle='-')
plt.text(0.2, 0.08, 'y=0.  '+'σ='+str(sigma), fontsize=14, color = "b", style = "italic")
#plt.title('q')
plt.xlabel('q',fontsize=14)
plt.ylabel('predict-q',fontsize=14)
plt.savefig('q.jpg')

plt.figure(3)
sigma = np.std(datar[0,:]/100-datar[1,:]/100)
sigma = round(sigma, 4)
plt.plot(datar[0,:]/100, datar[1,:]/100, '.', color='darkorange')
plt.plot(datar[0,:]/100, datar[0,:]/100, '-', c='black')
plt.plot(datar[0,:]/100, (datar[0,:]-datar[1,:])/100+0.35, '.', c='blue')
plt.axhline(y=0.35, color='r', linestyle='-')
#plt.title('r')
plt.text(0.4, 0.37, 'y=0.35  '+'σ='+str(sigma), fontsize=14, color = "b", style = "italic")
plt.xlabel('r',fontsize=14)
plt.ylabel('predict-r',fontsize=14)
plt.savefig('r.jpg')





