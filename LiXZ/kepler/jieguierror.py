# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 00:03:09 2020

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt

errordata = np.loadtxt('jieguierror.txt')

xuzhiq = errordata[:,0]
xuzhiincl = errordata[:,2]

dingincl = errordata[:,3]
dingq = errordata[:,1]

plt.figure(0)
sigma = np.std(dingq-xuzhiq)
sigma = round(sigma, 4)
plt.scatter(xuzhiq, dingq)
plt.plot(xuzhiq, xuzhiq)
plt.xlabel('q(Li et al.2020)',fontsize=14)
plt.ylabel('q(network)',fontsize=14)

plt.scatter(xuzhiq, dingq-xuzhiq)
plt.axhline(y=0, color='r', linestyle='-')

plt.text(0.12, 0.04, 'y=0  '+'σ='+str(sigma), fontsize=14, color = "b", style = "italic")

plt.figure(1)
sigma = np.std(dingincl-xuzhiincl)
sigma = round(sigma, 4)
plt.scatter(xuzhiincl, dingincl)
plt.plot(xuzhiincl, xuzhiincl)

plt.scatter(xuzhiincl, dingincl-xuzhiincl+65)
plt.axhline(y=65, color='r', linestyle='-')
plt.xlabel('incl(Li et al.2020)',fontsize=14)
plt.ylabel('incl(network)',fontsize=14)

plt.text(72, 66, 'y=65  '+'σ='+str(sigma), fontsize=14, color = "b", style = "italic")

print('ding-paper stdincl=', np.std(dingincl-xuzhiincl))

print('ding-paper stdq=', np.std(dingq-xuzhiq))