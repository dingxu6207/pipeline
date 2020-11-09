import phoebe
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm_notebook as tqdm


#warnings.filterwarnings('ignore')
logger = phoebe.logger(clevel = 'WARNING')

b = phoebe.default_binary(contact_binary=True)

times  = np.linspace(0,1,100)
b.add_dataset('lc', times=times)

b['period@binary'] = 1
b['sma@orbit'] = 1

m = 0
for mc in range(39,68):
    for j in range(100,6,-1):
        for i in range(10,90):
            for tem in range(11):          
                try:
                
                    print(mc,j,i,tem)
                    
                    b['requiv@primary'] = 0.01*mc
                    b['incl@binary'] = i
                    b['q@binary'] = 0.01*j
                    b['teff@primary'] = 5000-100*tem
                    b['teff@secondary'] = 5000
                    
                    b.run_compute(irrad_method='none')
                    print('it is ok1')
                
                    m = m+1
                    file = str(m)+'.lc'
                    lightcurvedata = np.vstack((b['value@times@lc01@model'], b['value@fluxes@lc01@model'])).T
                    mq = [(i, 0.01*j), (0.01*mc, 100*tem)]
                    datamq = np.array(mq)
                    print('it is ok2')
                
                    resultdata = np.row_stack((lightcurvedata, datamq))
                    np.savetxt(file, resultdata)
                    print('it is ok3')
                
                except:
                    print('it is error!')
                    
        
