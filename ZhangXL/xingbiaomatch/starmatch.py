# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 09:39:56 2020

@author: dingxu
"""

import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from photutils import CircularAperture, CircularAnnulus
from photutils import aperture_photometry
from astropy.time import Time
import astroalign as aa
import cv2
from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats

import pandas as pd
import astroalign as aa

def adjustimage(imagedata, coffe):
    mean = np.mean(imagedata)
    sigma = np.std(imagedata)
    mindata = np.min(imagedata)
    maxdata = np.max(imagedata)
    Imin = mean - coffe*sigma
    Imax = mean + coffe*sigma
        
    mindata = max(Imin,mindata)
    maxdata = min(Imax,maxdata)
    return mindata,maxdata


def displayimage(img, coff, i):
    minimg,maximg = adjustimage(img, coff)
    plt.figure(i)
    plt.imshow(img, cmap='gray', vmin = minimg, vmax = maximg)
    
  
def findsource(img):    
    mean, median, std = sigma_clipped_stats(img, sigma=3.0)
    daofind = DAOStarFinder(fwhm=4.52, threshold=5.*std)
    sources = daofind(img - median)

    for col in sources.colnames:
        sources[col].info.format = '%.8g'  # for consistent table output
        #print(sources)

    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    positionflux = np.transpose((sources['xcentroid'], sources['ycentroid'],  sources['flux']))
    mylist = positionflux.tolist()
    
    return positions,mylist
    
filename = 'E:\\shunbianyuan\\data\\origindata\\ftboYFAk010148.fits'
onehdu = fits.open(filename)
imgdata = onehdu[0].data  #hdu[0].header


positions1,mylist1 = findsource(imgdata)
apertures1 = CircularAperture(positions1, r=10.)
displayimage(imgdata, 1 ,0)
#apertures1.plot(color='blue', lw=1.5, alpha=0.5)
plt.plot(positions1[15][0], positions1[15][1], '*')

file = 'E:\\shunbianyuan\\phometry\\pipelinecode\\pipeline\\ZhangXL\\xingbiaomatch\\'+'dx2.csv'
df = pd.read_csv(file,sep=',')
npgaia = df.as_matrix()

transf, (s_list, t_list) = aa.find_transform(positions1, npgaia)



