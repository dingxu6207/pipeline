# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 16:14:03 2021

@author: dingxu
"""

from astropy.extern.six.moves.urllib import request
import tarfile
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

#url = 'http://python4astronomers.github.io/_downloads/core_examples.tar'
#tarfile.open(fileobj=request.urlopen(url), mode='r|').extractall()

hdus = fits.open('py4ast/core/3c120_stis.fits.gz')

primary = hdus[0].data  # Primary (NULL) header data unit
img = hdus[1].data      # Intensity data
err = hdus[2].data      # Error per pixel
dq = hdus[3].data       # Data quality per pixel

plt.figure(0)
plt.clf()
plt.imshow(img, origin = 'lower', vmin = -10, vmax = 65)
plt.colorbar()

plt.figure(1)  # Start a new plot -- by default matplotlib overplots.
plt.plot(img[:, 300])

profile = img.sum(axis=1)
plt.figure(2)
plt.plot(profile)

spectrum = img.sum(axis=0)
plt.figure(3)
plt.plot(spectrum)

plt.figure(4)
plt.clf()
plt.plot(img[:, 254:259])