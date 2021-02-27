# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 03:35:27 2021

@author: dingxu
"""

from astroquery.astrometry_net import AstrometryNet
from astropy.wcs import WCS
import numpy as np


path = "E:/shunbianyuan/phometry/todingx/origindata/"
file = "ftboYFAk300222.fits"
filename = path+file

#filename = 'new-image.fits'
ast = AstrometryNet()
ast.api_key = "vslojcwowmxjczlq"


wcs_header = ast.solve_from_image(filename, force_image_upload=True)

print(wcs_header)


wcs_gamcas = WCS(wcs_header)
print(wcs_gamcas)

pixcrd = np.array([[0, 0], [24, 38], [45, 98]], np.float_)
world = wcs_gamcas.wcs_pix2world(pixcrd, 1)
print(world)


'''
try_again = True
submission_id = None

while try_again:
    try:
        if not submission_id:
            wcs_header = ast.solve_from_image(filename,submission_id=submission_id)
        else:
            wcs_header = ast.monitor_submission(submission_id,solve_timeout=10*60*60)
    except TimeoutError as e:
        submission_id = e.args[1]
    else:
        # got a result, so terminate
        try_again = False


#when solve is ok, record data in wcs_gamcas object
if wcs_header:
    print('Solve OK - Print wcs result ...')
    wcs_gamcas = WCS(wcs_header)
    print(wcs_gamcas)
    
else:
    print('nok')
    
lon, lat = wcs_gamcas.all_pix2world(500, 500, 0)
print(lon, lat)
'''    