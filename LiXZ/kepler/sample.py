import phoebe
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm_notebook as tqdm
import random

#warnings.filterwarnings('ignore')
logger = phoebe.logger(clevel = 'WARNING')

b = phoebe.default_binary(contact_binary=True)

#b.filter(qualifier = 'l3_mode')

times  = np.linspace(0,1,100)
b.add_dataset('lc', times=times)

b['period@binary'] = 1
b['sma@orbit'] = 1

m = 0


for count in range(0,3):
    try:
        incl = random.uniform(70,90)
        T1divT2 = random.uniform(0.8,1.2)
        q = random.uniform(0.04,1)
        r = random.uniform(0.3,0.7)
        l3fraction = random.uniform(0,1)
        
        print('incl=', incl)
        print('temp=', 6500*T1divT2)
        print('q=', q)
        print('r=', r)
        print('i3fraction=', l3fraction)
        print('count = ', count)

        b['requiv@primary'] = r
        b['incl@binary'] = incl
        b['q@binary'] = q
        b['teff@primary'] = 6500
        b['teff@secondary'] = 6500*T1divT2
        b.set_value('l3_mode', 'fraction')
        b.set_value('l3_frac', l3fraction)

        #print(b.get_parameter('l3_frac').Value)

        b.run_compute(irrad_method='none')
        print('it is ok1')
    
        m = m+1
        file = str(m)+'.lc'
        lightcurvedata = np.vstack((b['value@times@lc01@model'], b['value@fluxes@lc01@model'])).T
        mq = [(incl, q), (r, T1divT2), (l3fraction, 0)]
        datamq = np.array(mq)
        print('it is ok2')
    
        resultdata = np.row_stack((lightcurvedata, datamq))
        np.savetxt(file, resultdata)
        
        print('it is ok3')
        
    except:
         print('it is error!')
        
        

