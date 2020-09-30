# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 21:37:57 2020

@author: dingxu
"""

import phoebe
from phoebe import u # units
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.logger()

b = phoebe.default_binary()

b.add_dataset('lc', times=np.linspace(0,1,201), dataset='mylc')

b.run_compute(irrad_method='none')

plt.figure(0)
afig, mplfig = b['mylc@model'].plot(show=True)


plt.figure(1)
afig, mplfig = b['mylc@model'].plot(x='phases', show=True)