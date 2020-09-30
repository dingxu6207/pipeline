# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:47:52 2020

@author: dingxu
"""

import phoebe
import numpy as np
import matplotlib.pyplot as plt
import emcee
import sys

b = phoebe.default_binary()

b['period@orbit'] = 0.5
b['sma@orbit'] = 3.5
b['incl@orbit'] = 83.5
b['requiv@primary'] = 1.2
b['requiv@secondary'] = 0.8
b['teff@primary'] = 6500.
b['teff@secondary'] = 5500.


b.add_dataset('lc', times=np.linspace(0, 0.5, 51))

b.run_compute()

b.plot(show=True)

np.savetxt('data.lc', 
           np.vstack((b['value@times@lc01@model'], b['value@fluxes@lc01@model']+np.random.normal(0, 0.01, 51))).T)

