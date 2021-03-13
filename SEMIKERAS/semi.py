# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 22:43:56 2020

@author: dingxu
"""

import phoebe
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.logger()

#b = phoebe.default_binary(contact_binary=True)
b = phoebe.default_binary()
b.add_constraint('semidetached', 'primary')

times  = np.linspace(0,1,100)
b.add_dataset('lc', times=phoebe.linspace(0,1,100))#compute_phases

#b.add_dataset('rv', times=phoebe.linspace(0,1,21), dataset='rv01')

b['period@binary'] = 1

b['incl@binary'] = 90
b['sma@binary'] = 1

b['q@binary'] = 0.4

b['teff@primary'] = 6000

b['teff@secondary'] = 5000

b['requiv@secondary'] = 0.2


b.add_dataset('mesh', times=[0.25], dataset='mesh01')

b.run_compute(irrad_method='none')

afig, mplfig = b.plot(show=True, legend=True)

lightdata = np.vstack((b['value@times@lc01@model'], b['value@fluxes@lc01@model'])).T

datamq = [(40,90), (77,77)]
npdata = np.array(datamq)

resultdata = np.row_stack((lightdata, npdata))

