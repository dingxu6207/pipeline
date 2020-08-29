# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 22:48:21 2020

@author: dingxu
"""

from astropy.io import fits
import numpy as np

wcshdu = fits.open('E:\\shunbianyuan\\phometry\\data\\'+'corr.fits')
table = wcshdu[1].data
print(wcshdu.info())
print(type(table))
nt = np.array(table)

a0 = table.field(2)
a1 = table.field(3)