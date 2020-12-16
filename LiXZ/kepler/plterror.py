# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:52:45 2020

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt

filetemp = 'E:\\shunbianyuan\\phometry\\pipelinecode\\pipeline\\LiXZ\\kepler\\data\\'


fileincl = 'incl.txt'

filedelT = 'delT.txt'

fileq = 'q.txt'

filer = 'r.txt'


dataincl = np.loadtxt(filetemp+fileincl)
#dataincl = dataincl.T

datadivt = np.loadtxt(filetemp+filedelT)

dataq = np.loadtxt(filetemp+fileq)

datar = np.loadtxt(filetemp+filer)

plt.figure(0)
#plt.plot(dataincl[0,:], dataincl[0,:]-dataincl[1,:], '.')
plt.plot(dataincl[0,:], dataincl[1,:], '.')
plt.title('incl')
plt.savefig('incl.jpg')

plt.figure(1)
plt.plot(datadivt[0,:], (datadivt[0,:]-datadivt[1,:])/100, '.')
plt.title('T1divT2')
plt.savefig('T1divT2.jpg')

plt.figure(2)
plt.plot(dataq[0,:], (dataq[0,:]-dataq[1,:])/100, '.')
plt.title('q')
plt.savefig('q.jpg')

plt.figure(3)
plt.plot(datar[0,:], (datar[0,:]-datar[1,:])/100, '.')
plt.title('r')
plt.savefig('r.jpg')





