# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 01:10:36 2021

@author: dingxu
"""

from astropy.wcs import WCS
from astropy.io import fits
import numpy as np

path = 'E:\\shunbianyuan\\phometry\\todingx\\origindata\\'
file = 'ftboYFAk300222.fits'
filename = path+file


#w = WCS(filename)

w = WCS('new-image.fits')

lon, lat = w.all_pix2world(1968.054, 1990.535, 0)

print(lon, lat)


pixcrd = np.array([[0, 0], [24, 38], [45, 98]], np.float_)
world = w.wcs_pix2world(pixcrd, 1)
print(world)