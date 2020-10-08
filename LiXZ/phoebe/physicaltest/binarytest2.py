# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 21:52:11 2020

@author: dingxu
"""

import phoebe
from phoebe import u # units
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.logger()

b = phoebe.default_binary(contact_binary=True)
#b = phoebe.default_binary()

b['q@binary'] = 0.4051

b['incl@binary'] = 78.48

print(b['mass@secondary@component']/b['mass@primary@component'])

#print(b.filter(component='primary', kind='star', context='component'))

#print(b.filter(component='secondary', kind='star', context='component'))

print(b.filter(context='component', kind='envelope'))

#print(b.filter(component='binary', context='component'))

b.add_dataset('mesh', compute_times=[0], dataset='mesh01')
b.add_dataset('orb', compute_times=np.linspace(0,1,201), dataset='orb01')
b.add_dataset('lc', times=np.linspace(0,1,21), dataset='lc01')
b.add_dataset('rv', times=np.linspace(0,1,21), dataset='rv01')

b.run_compute(irrad_method='none')

print(b['mesh01@model'].components)

plt.figure(0)
afig, mplfig = b['mesh01@model'].plot(x='ws', show=True)

plt.figure(1)
afig, mplfig = b['orb01@model'].plot(x='ws',show=True)

plt.figure(2)
afig, mplfig = b['lc01@model'].plot(show=True)
     
plt.figure(3)         
afig, mplfig = b['rv01@model'].plot(show=True)