# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 03:35:27 2021

@author: dingxu
"""

from astroquery.astrometry_net import AstrometryNet


path = "E:/shunbianyuan/phometry/todingx/origindata/"
file = "ftboYFAk300222.fits"
filename = path+file

ast = AstrometryNet()
ast.api_key = "vslojcwowmxjczlq"

wcs_header = ast.solve_from_image(filename)

