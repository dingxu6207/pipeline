# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 20:43:35 2021

@author: dingxu
"""

import phoebe
import numpy as np
import matplotlib.pyplot as plt

b = phoebe.default_binary()
b.add_constraint('semidetached', 'primary')

print(b['requiv@constraint@primary'])

b['q'] = 1.3

b['sma@binary'] = 6.

b.add_dataset('mesh', times=[0.25])

b.run_compute(irrad_method='none')

afig, mplfig = b.plot(show=True)

print(b.filter(context='component', kind='star', component='primary'))

#print(b.filter(kind='envelope'))
print(b['requiv@secondary'])