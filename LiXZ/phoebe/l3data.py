# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 23:20:18 2020

@author: dingxu
"""

import phoebe
from phoebe import u # units
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.logger()

b = phoebe.default_binary()

#b.filter(qualifier='l3_mode')

b.add_dataset('lc', times=np.linspace(0,1,101), dataset='lc01')


#b.run_compute(irrad_method='none', model='no_third_light')
b.set_value('l3_mode', 'fraction')
b.set_value('l3_frac', 0.0)
#b.set_value('l3_mode', 'flux')
#b.set_value('l3', 5)

print(b.filter(qualifier='l3*'))
#print(b.get_parameter('l3'))

#b.run_compute(irrad_method='none', model='with_third_light')

#flux = b['times@with_third_light@model']
#time = b['fluxes@with_third_light@model']

b.run_compute(irrad_method='none')
print(b['value@times@latest@model'])
print(b['value@fluxes@latest@model'])

plt.plot(b['value@times@lc01@model'], b['value@fluxes@lc01@model'])
