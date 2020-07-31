# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:09:06 2020

@author: dingxu
"""
import numpy as np
import matplotlib.pylab as plt
# Import pyTiming
from PyAstronomy.pyTiming import pyPeriod

file = 'E:\\shunbianyuan\\pdm\\00\\00\\data.txt'

data = np.loadtxt(file)

qtime = data[:,0:1]
time = qtime- np.floor(np.min(qtime)) 
#time = np.arange(0,138,1)
flux = data[:,1:2]

'''
lc = pyPeriod.TimeSeries(time, flux)
# Compute the Leahy-normalized Fourier transform,
# plot the time series, and check that the mean
# power level is 2 as expected.
fft = pyPeriod.Fourier(lc)

m = np.where(fft.power == np.max(fft.power))
index = m[0][0]
print(fft.freq[index])

plt.figure(0)
plt.plot(fft.freq, fft.power)
'''

plt.figure(2)
plt.plot(time, flux, 'bp')