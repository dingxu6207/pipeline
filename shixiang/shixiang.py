# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 21:38:12 2020

@author: dingxu
"""

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import signal
import pandas as pd

path = 'E:\\shunbianyuan\\phometry\\data\\'

''' 
file = 'med-58025-HIP507401_sp02-003.fits.gz'
filename = path+file

hdulist = fits.open(filename)
hdulist.info()
#print(hdulist[0].header)
RV = hdulist[0].header['HELIO_RV']
print(RV)

i = 3
col = hdulist[i].columns

print (col.names)




tbdata = hdulist[i].data
fluxdata = tbdata['FLUX']
wave = tbdata['LOGLAM']
    
wavelist = 10**wave

plt.plot(wavelist, fluxdata)
'''

file = 'dr6_med_v1.1_MRS.csv'
filename = path+file

csv_data = pd.read_csv(filename,encoding='utf-8',header=None,sep = None)