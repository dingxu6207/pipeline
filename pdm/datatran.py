# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:38:03 2020

@author: dingxu
"""

from PyAstronomy import pyasl
import datetime as dt
from astropy.time import Time

# Convert July 14th, 2018, 10pm to a Julian date
d = dt.datetime(2018, 7, 14, 22, 10, 11)
jd = pyasl.jdcnv(d)

times = ['2018-07-14T22:10:11']
t = Time(times, format='isot', scale='utc')
print(jd)
print(t.jd[0])
print(t.mjd[0])
