# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 23:30:07 2020

@author: dingxu
"""

import phoebe
from phoebe import u
import numpy as np
import matplotlib.pyplot as plt


#b = phoebe.default_binary(contact_binary=True)
b = phoebe.default_binary()

b.add_constraint('semidetached', 'primary')

#b['period@binary'] = 1
b['sma@binary'] = 6.

b['q'] = 1.3

times = np.linspace(0,1,21)

b.add_dataset('lc', times=times, dataset='lc01')

#b.add_dataset('rv', times=times, dataset='rv01')

#b.add_dataset('mesh', times=times, columns=['visibilities', 'intensities@lc01','rvs@rv01' ], dataset='mesh01' )

b.run_compute(irrad_method='none')

'''
b['lc01@model'].plot(axpos=221)
b['rv01@model'].plot(c={'primary': 'blue', 'secondary': 'red'}, linestyle='solid', axpos=222)
b['mesh@model'].plot(fc='intensities@lc01', ec='None', axpos=425)
b['mesh@model'].plot(fc='rvs@rv01', ec='None', axpos=427)
b['mesh@model'].plot(fc='visibilities', ec='None', y='ws', axpos=224)

fig = plt.figure(figsize=(11,4))
afig, mplanim = b.savefig('animation_binary_complete.gif', fig=fig, tight_layouot=True, draw_sidebars=False, animate=True, save_kwargs={'writer': 'imagemagick'})
'''