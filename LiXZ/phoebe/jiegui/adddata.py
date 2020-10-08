# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 19:47:07 2020

@author: dingxu
"""

import phoebe
import numpy as np
import matplotlib.pyplot as plt
import emcee
import sys

#b = phoebe.default_binary(contact_binary=True)
b = phoebe.default_binary()


b['period@orbit'] = 0.5
b['sma@orbit'] = 3.5
b['incl@orbit'] = 83.5
b['requiv@primary'] = 1.2
b['requiv@secondary'] = 0.8
b['teff@primary'] = 6500.
b['teff@secondary'] = 5500.

times = np.linspace(0, 1.5, 100) 
b.add_dataset('lc', times=np.linspace(0, 1.5, 100), dataset='lc01')

print(b['value@fluxes@dataset'])
#b.add_dataset('lc', compute_phases = np.linspace(0,1,100), dataset='lc01')

b.run_compute()



np.savetxt('data.lc', 
           np.vstack((b['value@times@lc01@model'], b['value@fluxes@lc01@model']+np.random.normal(0, 0.01, 100))).T)

b.flip_constraint('compute_phases', 'compute_times')
b['compute_phases@lc@dataset'] = np.linspace(-0.5,0.5,21)

phases = b.to_phase(times)
phases_sorted = sorted(phases)


plt.figure(0)
b.plot(show=True)

plt.figure(1)
flux = b['fluxes@model'].interp_value(phases=phases_sorted)
plt.plot(phases_sorted,flux)

#plt.plot(phases_sorted,flux,'.')
