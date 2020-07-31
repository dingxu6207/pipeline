# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:53:14 2020

@author: dingxu
"""


import matplotlib.pylab as plt
import numpy as np



# Generate some data ...
PATH = 'E:\\shunbianyuan\\phometry\\countperiod\\star\\'
data = np.loadtxt(PATH+'datamag.txt')


time = data[0:1,:]
time = time.T

flux = data[1:2,:]
flux = flux.T
'''
time = np.random.random(1000)*100.
flux = 0.05*np.sin(time*(2.*np.pi/21.5) + 15)
# ... and add some noise
flux += np.random.normal(0, 0.02, len(flux))
'''

plt.figure(0)
plt.plot(time, flux, 'bp')
ax = plt.gca()
ax.yaxis.set_ticks_position('right') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
# Obtain the phases with respect to some
# reference point (in this case T0=217.4)
t = 1.5
phasedata = []
fluxdata = []
for i in range(len(time)):
    pd = time[i]/t - int(time[i]/t)
    phasedata.append(pd)
    fluxdata.append(flux[i])
    
phase = np.array(phasedata) 
flux = np.array(fluxdata)



plt.figure(1)
plt.plot(phase, flux, 'bp')
ax = plt.gca()
ax.yaxis.set_ticks_position('right') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
