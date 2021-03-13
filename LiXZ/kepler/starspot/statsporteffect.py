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
cb['q'] = 0.7
cb['period@binary'] = 1
#cb['fillout_factor@contact_envelope@envelope@component'] = 0.5
#cb.run_compute(irrad_method='none')

#cb.add_feature('spot', component='primary', feature='spot01', relteff=0.8, radius=23, colat=90, long=-45, overwrite=True)
cb.add_dataset('mesh', times=[0.25], dataset='mesh01', columns=['teffs'], overwrite=True)
cb.add_dataset('lc', times=np.linspace(0.,1,150), dataset='lc01', overwrite=True)

cb.run_compute(irrad_method='none', model='with_spot', overwrite=True)

plt.figure(0)
#axs, artists = cb.plot('mesh01', fc='teffs', ec='face', fcmap='plasma', show=True)
afig, mplfig = cb.plot('mesh01@with_spot', fc='teffs', ec='face', fcmap='plasma', show=True)

plt.figure(1)
#axs, artists = cb.plot('lc01', show=True)
#afig, mplfig = cb.plot('lc', show=True, legend=True)
times = cb['value@times@lc01@with_spot@model']  #b['value@times@lc01@model']
flux =  cb['value@fluxes@with_spot@model']
#print(cb.filter(context='dataset'))
mag = -2.5*np.log(flux)
plt.plot(times, mag)

cb.add_feature('spot', component='primary', feature='spot01', relteff=0.8, radius=23, colat=90, long=-90, overwrite=True)
cb.run_compute(irrad_method='none', model='with_spot', overwrite=True)

times = cb['value@times@lc01@with_spot@model']  #b['value@times@lc01@model']
flux =  cb['value@fluxes@with_spot@model']
#print(cb.filter(context='dataset'))
mag = -2.5*np.log(flux)
plt.plot(times, mag)

ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向

plt.xlabel('phase',fontsize=14)
plt.ylabel('mag',fontsize=14)

plt.legend(('no spot', 'with spot'), loc='upper right')

plt.figure(2)
#axs, artists = cb.plot('mesh01', fc='teffs', ec='face', fcmap='plasma', show=True)
afig, mplfig = cb.plot('mesh01@with_spot', fc='teffs', ec='face', fcmap='plasma', show=True)