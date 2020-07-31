# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 17:14:03 2020

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = np.loadtxt('pms.txt')


arraypms = data
ERROR = 1/arraypms[:,3]   

plt.figure(0)
plt.plot(arraypms[:,2], ERROR ,'.')  
plt.xlabel('mag',fontsize=14)
plt.ylabel('error',fontsize=14)  

def fund(x, a, b):
    return a*(np.exp(b*x))

x = arraypms[:,2]
popt, pcov = curve_fit(fund, x, ERROR)

#ydata = popt[0]*x**popt[1]
ydata = popt[0]*(np.exp(popt[1]*x))
plt.figure(1)
plt.plot(x,ydata,'rp')
plt.plot(x,ERROR,'bp')

print(popt[0], popt[1])
