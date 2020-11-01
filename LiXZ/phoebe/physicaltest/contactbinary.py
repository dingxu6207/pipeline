# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 06:37:29 2020

@author: dingxu
"""

import phoebe
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.logger()

b = phoebe.default_binary(contact_binary=True)

times  = np.linspace(0,1,100)
b.add_dataset('lc', times=phoebe.linspace(0,1,100))#compute_phases
#b.add_dataset('lc', compute_phases=phoebe.linspace(0,1,101))

b['period@binary'] = 1

b['incl@binary'] = 90

b['q@binary'] = 0.46

b['teff@primary'] = 4080

b['teff@secondary'] = 4000

#b['fillout_factor@contact_envelope@envelope@component'] = 0.5

b['sma@binary'] = 1 #0.05 2.32
#print(b['sma@binary'])

b['requiv@primary'] = 0.52 #(0.6,0.68)-(0.03,0.11) 

#print(b['requiv@primary'])
#print(b['requiv@secondary'])

b.add_dataset('mesh', times=[0.25], dataset='mesh01')

b.run_compute(irrad_method='none')

plt.figure(0)
afig, mplfig = b.plot(show=True, legend=True)

print(b['fillout_factor@contact_envelope'])



np.savetxt('data0.lc', 
           np.vstack((b['value@times@lc01@model'], b['value@fluxes@lc01@model'])).T)


'''
phases = b.to_phase(times)
phases_sorted = sorted(phases)
flux = b['fluxes@model'].interp_value(phases=phases_sorted)
'''
'''
from PyAstronomy.pyasl import foldAt
phases = foldAt(times, 3)
sortIndi = np.argsort(phases)
# ... and, second, rearrange the arrays.
phases = phases[sortIndi]
flux = b['value@fluxes@lc01@model'][sortIndi]

np.savetxt('data1.lc', np.vstack((phases, flux)).T)

plt.figure(1)
plt.plot(phases, flux)
'''

'''
for i in range(4):
    b['sma@binary'] = 3.1+0.1*i
    b.run_compute(irrad_method='none')
    plt.figure(i)
    b.plot(show=True, legend=True)
    print(b['fillout_factor@contact_envelope'])

'''