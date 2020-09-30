# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 19:47:07 2020

@author: dingxu
"""

import phoebe
import numpy as np
import matplotlib.pyplot as plt
import emcee
import sys

b = phoebe.default_binary()

'''
b['period@orbit'] = 0.5
b['sma@orbit'] = 3.5
b['incl@orbit'] = 83.5
b['requiv@primary'] = 1.2
b['requiv@secondary'] = 0.8
b['teff@primary'] = 6500.
b['teff@secondary'] = 5500.
b['q@binary@orbit@component'] = 0.7
'''

b['period@orbit'] = 1.5
b['incl@binary@orbit@component'] = 86.5
b['requiv@primary@star@component'] = 1.2
b['requiv@secondary@star@component'] = 0.8
b['q@binary@orbit@component'] = 0.7
#b['pblum@primary@dataset'] = 2.9*np.pi


b.add_dataset('lc', times=np.linspace(0, 3, 151))

b.run_compute()

b.plot(show=True)

np.savetxt('data.lc', 
           np.vstack((b['value@times@lc01@model'], b['value@fluxes@lc01@model']+np.random.normal(0, 0.01, 151))).T)

