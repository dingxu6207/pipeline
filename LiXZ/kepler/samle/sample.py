import phoebe
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm_notebook as tqdm
import random

#warnings.filterwarnings('ignore')
logger = phoebe.logger(clevel = 'WARNING')

b = phoebe.default_binary(contact_binary=True)

times  = np.linspace(0,1,100)
b.add_dataset('lc', times=times)

b['period@binary'] = 1
b['sma@orbit'] = 1

m = 0


for count in range(0,500000):
    try:
        incl = random.uniform(30,90)
        T1divT2 = random.uniform(0.8,1.2)
        #q1 = random.uniform(3,10)
        q = np.random.normal(loc=0, scale=0.5)
        #q1 = np.random.normal(loc=5, scale=2)
        #q = np.concatenate((q,q1))
        q = np.abs(q)       
        r = random.uniform(0.7,0.3)
        
        print('incl=', incl)
        print('temp=', 6500*T1divT2)
        print('q=', q)
        print('r=', r)
        print('count = ', count)
        
        b['requiv@primary'] = r
        b['incl@binary'] = incl
        b['q@binary'] = q
        b['teff@primary'] = 6500
        b['teff@secondary'] = 6500*T1divT2
        '''
        b.run_compute(irrad_method='none')
        print('it is ok1')
    
        m = m+1
        file = str(m)+'.lc'
        lightcurvedata = np.vstack((b['value@times@lc01@model'], b['value@fluxes@lc01@model'])).T
        mq = [(incl, q), (r, T1divT2)]
        datamq = np.array(mq)
        print('it is ok2')
    
        resultdata = np.row_stack((lightcurvedata, datamq))
        np.savetxt(file, resultdata)
        '''
        print('it is ok3')
        
    except:
         print('it is error!')
        
        

