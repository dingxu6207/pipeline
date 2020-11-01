# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:01:21 2020

@author: dingxu
"""

import phoebe
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.logger()

b = phoebe.default_binary(contact_binary=True)

times  = np.linspace(0,1,200)
b.add_dataset('lc', times=times)

b['period@binary'] = 1
b['sma@orbit'] = 1

for mc in range(100):
    for j in range(100):
        for i in range(90):
            try:
                
                b['requiv@primary'] = 0.01*mc
                b['incl@binary'] = i
                b['q@binary'] = 0.01*j
                b.run_compute(irrad_method='none')
                print('it is ok1')
                
                m = i*10+j
                file = str(m)+'.lc'
                lightcurvedata = np.vstack((b['value@times@lc01@model'], b['value@fluxes@lc01@model'])).T
                mq = [(b['incl@binary'], b['q@binary']), (b['requiv@primary'], 0)]
                datamq = np.array(mq)
                print('it is ok2')
                
                resultdata = np.row_stack((lightcurvedata, datamq))
                np.savetxt(file, resultdata)
                print('it is ok3')
                
            except:
                print('it is error!')
        


        

