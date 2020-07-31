# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 21:24:16 2020

@author: dingxu
"""

from astropy.time import Time
from astropy.io import fits
import os
import numpy as np

filetemp = []
count = 0
oripath = 'E:\\shunbianyuan\\dataxingtuan\\ngc7423\\'
for root, dirs, files in os.walk(oripath):
   for file in files:
       if (file[-4:] == '.fit'):
       #print(file)
           count = count+1
           filetemp.append(file)



#2019-11-15T02:49:09
fitsname1 = oripath+filetemp[200]
onehdu = fits.open(fitsname1)
DATEOBS = onehdu[0].header['DATE-OBS']
TIME = onehdu[0].header['TIME']
imgdata1 = onehdu[0].data  #hdu[0].header
copydata1 = np.copy(imgdata1)   
print(DATEOBS+TIME)

datatime = '20'+DATEOBS[-2:]+'-'+DATEOBS[-5:-3]+'-'+DATEOBS[-8:-6]+'T'+TIME[-10:]

t = Time(datatime, format='isot', scale='utc')
print(t.jd)

 