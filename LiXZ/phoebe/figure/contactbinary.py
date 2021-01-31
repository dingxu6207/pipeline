# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 06:37:29 2020

@author: dingxu
"""

import phoebe
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.logger()

#b = phoebe.default_binary(contact_binary=True)

b = phoebe.default_binary()


times  = np.linspace(0,1,150)

#b.add_dataset('lc', times=phoebe.linspace(0,1,100), passband= 'Kepler:mean')#compute_phases
b.add_dataset('lc', times=phoebe.linspace(0,2,150))

b['period@binary'] = 2

b['incl@binary'] =  90 #58.528934
b['q@binary'] =  27*0.01
b['teff@primary'] =  6500  #6208 
b['teff@secondary'] = 6500*90*0.01#6500*100.08882*0.01 #6087


#b['fillout_factor@contact_envelope@envelope@component'] = 0.5

b['sma@binary'] = 4#0.05 2.32
#print(b['sma@binary'])

b['requiv@primary'] = 50*0.01    #0.61845703

b['requiv@secondary'] = 50*0.01 

b.add_dataset('mesh', times=[0.25])

b.run_compute(irrad_method='none')

plt.figure(0)
afig, mplfig = b.plot(show=True, legend=True)

print(b['fillout_factor@contact_envelope'])



np.savetxt('detach.lc', 
           np.vstack((b['value@times@lc01@model'], b['value@fluxes@lc01@model'])).T)

