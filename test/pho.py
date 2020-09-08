# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:48:56 2020

@author: dingxu
"""
from photutils import CircularAperture, CircularAnnulus
from photutils.datasets import make_100gaussians_image
import matplotlib.pyplot as plt

positions = [(145.1, 168.3), (84.5, 224.1), (48.3, 200.3)]
aperture = CircularAperture(positions, r=5)
annulus_aperture = CircularAnnulus(positions, r_in=10, r_out=15)
annulus_masks = annulus_aperture.to_mask(method='center')
data = make_100gaussians_image()
annulus_data = annulus_masks[0].multiply(data)
plt.imshow(annulus_data)
plt.colorbar()