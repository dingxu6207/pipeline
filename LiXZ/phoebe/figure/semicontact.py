# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 17:40:12 2021

@author: dingxu
"""

import phoebe
import numpy as np
import matplotlib.pyplot as plt

b = phoebe.default_binary()


b.add_constraint('semidetached', 'primary')
b['sma@binary'] = 6.

b['period@binary'] = 1

b['incl@binary'] =  90 #58.528934
b['q@binary'] =  27*0.01
b['teff@primary'] =  6500  #6208 
b['teff@secondary'] = 6500*90*0.01#6500*100.08882*0.01 #6087


#b['requiv@primary'] = 50*0.01    #0.61845703

#b['requiv@secondary'] = 50*0.01 


b.add_dataset('lc', times=phoebe.linspace(0,1,100))

b.add_dataset('mesh', times=[0.25])

b.run_compute(irrad_method='none')

afig, mplfig = b.plot(show=True)


np.savetxt('semi1.lc', 
           np.vstack((b['value@times@lc01@model'], b['value@fluxes@lc01@model'])).T)



