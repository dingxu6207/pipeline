# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:52:45 2020

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt

filetemp = 'E:\\shunbianyuan\\phometry\\pipelinecode\\pipeline\\LiXZ\\kepler\\data\\'


fileincl = 'onlyincl.txt'


dataincl = np.loadtxt(filetemp+fileincl)
#dataincl = dataincl.T

sigma = np.std(dataincl[0,:]-dataincl[1,:])
sigma = round(sigma, 3)

plt.figure(0)
plt.plot(dataincl[0,:], dataincl[0,:]-dataincl[1,:]+10, '.')
plt.plot(dataincl[0,:], dataincl[1,:], '.')
plt.plot(dataincl[0,:], dataincl[0,:], '-', c='r')
plt.axhline(y=10, color='r', linestyle='-')
plt.text(45, 15, 'y=10  '+'σ='+str(sigma), fontsize=14, color = "b", style = "italic")
#plt.title('incl', fontsize=20)
plt.xlabel('incl',fontsize=14)
plt.ylabel('predict-incl',fontsize=14)
plt.savefig('incl.jpg')

