# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 21:52:34 2020

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt
import phoebe

data1 = np.loadtxt('data0.lc')

data2 = np.loadtxt('data1.lc')

plt.figure(0)
plt.plot(data1[:,0], data1[:,1], '.')
plt.plot(data2[:,0], data2[:,1], '.')

logger = phoebe.logger()

b = phoebe.default_binary(contact_binary=True)

times  = data1[:,0]
#b.add_dataset('lc', times=phoebe.linspace(0,1,100))#compute_phases
b.add_dataset('lc', compute_phases = data1[:,0] ,fluxes =  data1[:,1])

b.set_value('period', component='binary', value=3.14)

#print(b.filter(qualifier=['compute_times', 'compute_phases']))


print(b['value@compute_times@dataset'], b['value@fluxes@dataset'])
plt.figure(1)
plt.plot(b['value@compute_times@dataset'], b['value@fluxes@dataset'])
