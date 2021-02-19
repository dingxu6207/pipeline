# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 22:27:34 2021

@author: dingxu
"""

import phoebe
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.logger(clevel='WARNING')

cb = phoebe.default_binary(contact_binary = True)

#print(cb.filter(context='component', kind='star', component='primary'))
#
#print(cb.filter(context='component', kind='star', component='secondary'))
#
#print(cb['requiv@secondary@constraint'])
#
#print(cb.filter(context='component', kind='envelope'))


#print(cb.filter(context='component')) #component='primary'


#print(cb.filter(component='binary')) 
cb['q'] = 1.
#cb['fillout_factor@contact_envelope@envelope@component'] = 0.5
#cb.run_compute(irrad_method='none')

cb.add_feature('spot', component='primary', feature='spot01', relteff=0.8, radius=23, colat=90, long=-45, overwrite=True)
cb.add_dataset('mesh', times=[0.125], dataset='mesh01', columns=['teffs'], overwrite=True)
cb.add_dataset('lc', times=np.linspace(0.,0.5,50), dataset='lc01', overwrite=True)

cb.run_compute(irrad_method='none', model='with_spot', overwrite=True)

plt.figure(0)
#axs, artists = cb.plot('mesh01', fc='teffs', ec='face', fcmap='plasma', show=True)
afig, mplfig = cb.plot('mesh01@with_spot', fc='teffs', ec='face', fcmap='plasma', show=True)

plt.figure(1)
#axs, artists = cb.plot('lc01', show=True)
afig, mplfig = cb.plot('lc', show=True, legend=True)

#print(cb['fluxes@model'])

times = cb['times@lc01@with_spot@model']
flux =  cb['fluxes@with_spot@model']
print(cb.filter(context='dataset'))