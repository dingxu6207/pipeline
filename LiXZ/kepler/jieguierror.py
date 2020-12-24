# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 00:03:09 2020

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt

errordata = np.loadtxt('jieguierror.txt')

xuzhiq = errordata[:,0]
xuzhiincl = errordata[:,1]

dingincl = errordata[:,2]
dingq = errordata[:,3]/100

plt.figure(0)
plt.subplot(211)
plt.scatter(xuzhiq, dingq)
plt.plot(xuzhiq, xuzhiq)
plt.xlabel('paper-q',fontsize=14)
plt.ylabel('ding-q',fontsize=14)

plt.subplot(212)
plt.scatter(xuzhiq, dingq-xuzhiq)
plt.plot(xuzhiq,xuzhiq-xuzhiq)
plt.xlabel('paper-q',fontsize=14)
plt.ylabel('delq',fontsize=14)


plt.figure(1)
plt.subplot(211)
plt.scatter(xuzhiincl, dingincl)
plt.plot(xuzhiincl, xuzhiincl)
plt.xlabel('paper-incl',fontsize=14)
plt.ylabel('ding-incl',fontsize=14)

plt.subplot(212)
plt.scatter(xuzhiincl, dingincl-xuzhiincl)
plt.plot(xuzhiincl,xuzhiincl-xuzhiincl)
plt.xlabel('paper-incl',fontsize=14)
plt.ylabel('delincl',fontsize=14)

print('ding-paper stdincl=', np.std(dingincl-xuzhiincl))

print('ding-paper stdq=', np.std(dingq-xuzhiq))