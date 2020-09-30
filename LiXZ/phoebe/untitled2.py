# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 21:41:43 2020

@author: dingxu
"""

import phoebe
from phoebe import u # units
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.logger()

#b = phoebe.default_contact_binary()
b = phoebe.default_binary()



times = np.linspace(0,1,51)



#b.set_value('q', value=0.75)


b['period@orbit'] = 0.5
b['sma@orbit'] = 3.5
b['incl@orbit'] = 83.5
b['requiv@primary'] = 1.2
b['requiv@secondary'] = 0.8
b['teff@primary'] = 6500.
b['teff@secondary'] = 5500.


b.add_feature('spot', component='primary', feature='spot01')

b.add_dataset('lc', compute_times=times, dataset='lc01')

b.add_dataset('orb', compute_times=times, dataset='orb01')

b.add_dataset('rv', times=times, dataset='rv01')

b.add_dataset('mesh', compute_times=times, dataset='mesh01', columns=['teffs'], overwrite=True)



b.run_compute(irrad_method='none')

afig, mplanim = b.plot(y={'orb': 'ws'}, 
                       animate=True, save='animations_1.gif', save_kwargs={'writer': 'imagemagick'})


