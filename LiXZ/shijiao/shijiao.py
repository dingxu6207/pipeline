# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 14:11:40 2020
https://docs.astropy.org/en/stable/coordinates/index.html
@author: dingxu
"""

from astropy import units as u
from astropy.coordinates import SkyCoord

jra = 431.000548
jdec = 2116.343129
c1 = SkyCoord(ra=10.625*u.degree, dec=41.2*u.degree, frame='icrs')
#c2 = SkyCoord(10.625, 41.2, frame='icrs', unit='deg')
c2 = SkyCoord(jra, jdec, frame='icrs', unit='deg')
print('c2.ra.hms=', c2.ra.hms)
print('c2.dec=', c2.dec)


#07:42:32.797
#+49:48:34.74
RA = '23h18m59.938s'  #21:50:56.794   21 45 10.0 +65 46 18
DEC = '+72d18m24.19s' #65:15:54.94
c3 = SkyCoord(RA, DEC, frame='icrs')

print('c3 = ', c3)

'''
print('c3.ra = ', c3.ra)
print('c3.ra.hms=', c3.ra.hms)

print('c3.dec=', c3.dec)
print('c3.dec.degree=', c3.dec.degree)
'''


