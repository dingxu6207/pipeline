# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 01:19:28 2020

@author: dingxu
"""

import pandas as pd  
import numpy as np   
from astropy.coordinates import SkyCoord   
                
data = pd.read_csv('NGC7142.csv')

dataradec = data['Coords']

listdata = dataradec.tolist()

#intlist = int(listdata[0][:2])
print(listdata[0])

RA = '23h21m02.95s'  #21:50:56.794   21 45 10.0 +65 46 18
DEC = '+71d49m20.0s' #65:15:54.94

RA = 

c3 = SkyCoord(RA, DEC, frame='icrs')

print('c3.dec.degree=', c3.dec.degree)
print('c3.ra.degree=', c3.ra.degree)

