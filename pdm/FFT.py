# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:24:54 2020

@author: dingxu
"""

import numpy as np
import matplotlib.pylab as plt
from scipy.fftpack import fft

PATH = 'E:\\shunbianyuan\\phometry\\countperiod\\star\\'
# Create some evenly sampled artificial data (Poisson noise)
data = np.loadtxt(PATH+'datamag.txt')
#time = np.arange(0,119,0.1)


time = data[0:1,:]
time = time.T

flux = data[1:2,:]
flux = flux.T
'''
time = np.arange(1000)/10
flux = 0.05*np.sin(time*(2.*np.pi/21.5) + 15)
# ... and add some noise
flux += np.random.normal(0, 0.02, len(flux))
'''
yy = fft(flux) 
yreal = yy.real
yimag = yy.imag 

yf=abs(fft(flux))


fs = 1/(time[2]-time[1])

xf = fs*np.arange(len(time))/len(time)        # 频率
xf1 = xf


plt.subplot(211)
plt.plot(time, flux, 'bp')   
plt.title('Original wave')

plt.subplot(212)
plt.plot(xf, yf,'r')
plt.title('FFT of Mixed wave(two sides frequency range)',fontsize=7,color='#7A378B')  #注意这里的颜色可以查询颜色代码表



