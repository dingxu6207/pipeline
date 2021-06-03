# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 22:06:25 2021

@author: dingxu
"""

import phoebe
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.logger()

b = phoebe.default_binary(contact_binary=True)

times  = np.linspace(0,1,150)

b.add_dataset('lc', times = times)

b['period@binary'] = 1

b['incl@binary'] =  55.0 
#b['q@binary'] =     0.4
b['teff@primary'] =  6500  
b['teff@secondary'] = 6500


#b['fillout_factor@contact_envelope@envelope@component'] = 0.1

b['sma@binary'] = 1#0.05 2.32
b['requiv@primary'] = 0.5    #0.61845703
b['requiv@secondary'] = 0.5    #0.61845703

print(b.get_constraint(qualifier='requiv', component='secondary'))
#b['fillout_factor@contact_envelope@envelope@component'] = 0.3
#b.get_constraint(qualifier='fillout_factor@contact_envelope@envelope@component')
'''
b['fillout_factor@contact_envelope@envelope@component'] = 0.3
'''
#b.flip_constraint(qualifier='fillout_factor', solve_for='q@primary')

#b['fillout_factor@contact_envelope'] = 0.4

#b.get_constraint(qualifier='fillout_factor')
#b.flip_constraint(qualifier='requiv@secondary@component', solve_for='requiv@primary@component')
#b.flip_constraint(qualifier='fillout_factor', context='contact_envelope', solve_for='requiv@primary@star@component')
#b['fillout_factor'] = 0.3

b.add_dataset('mesh', times=[0.25], dataset='mesh01')

b.run_compute(ntriangles=5000)

plt.figure(0)
afig, mplfig = b.plot(show=True, legend=True)

print(b['fillout_factor@contact_envelope'])

