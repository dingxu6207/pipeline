# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 21:17:14 2020

@author: dingxu
"""

import phoebe
from phoebe import u # units
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.logger()

b_cb = phoebe.default_binary(contact_binary=True)

b_detached = phoebe.default_binary()

print(b_detached.hierarchy)

print(b_cb.hierarchy)

#print(b_cb.filter(component='contact_envelope', kind='envelope', context='component'))

print(b_cb.filter(component='primary', kind='star', context='component'))


#print(b_cb.filter(component='binary'))

b_cb['requiv@primary'] = 1.5

b_cb['pot@contact_envelope@component']

