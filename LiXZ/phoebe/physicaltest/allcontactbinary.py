# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 07:21:15 2020

@author: dingxu
"""

import phoebe
import numpy as np



b = phoebe.default_binary()

b.add_constraint('semidetached', 'primary')
b.add_constraint('semidetached', 'secondary')

b['q@binary'] = 0.44
b['incl@binary'] = 78

b.add_dataset('lc', times=np.linspace(0,1,101))

b.add_dataset('mesh', times=[0.25])

b.run_compute(irrad_method='none')

afig, mplfig = b.plot(show=True)