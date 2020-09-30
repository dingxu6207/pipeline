# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:56:04 2020

@author: dingxu
"""

import numpy as np
import corner
import matplotlib.pyplot as plt

ndim, nsamples = 5, 10000
samples = np.random.randn(ndim * nsamples).reshape([nsamples, ndim])
figure = corner.corner(samples)
figure.savefig("corner.png")

