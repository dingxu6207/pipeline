# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:34:01 2020

@author: dingxu
"""

import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.time import Time

from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from photutils import CircularAperture

filetemp = []
count = 0
oripath = 'E:\\shunbianyuan\\Asteroids_Dingxu\\6478\\20200826_6478\\alligen\\'  #路径参数
for root, dirs, files in os.walk(oripath):
   for file in files:
       if (file[-4:] == '.fit'):
           count = count+1
           filetemp.append(file)
           
          
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



def findsource(img):    
    mean, median, std = sigma_clipped_stats(img, sigma=3.0)
    daofind = DAOStarFinder(fwhm = 4.52, threshold=5.*std)
    sources = daofind(img - median)

    for col in sources.colnames:
        sources[col].info.format = '%.8g'  # for consistent table output
        #print(sources)

    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    positionflux = np.transpose((sources['xcentroid'], sources['ycentroid'],  sources['flux']))
    mylist = positionflux.tolist()
    
    return sources,positions,mylist

changefile = 'E:\\shunbianyuan\\Asteroids_Dingxu\\6478\\20200826_6478\\location\\'
for i in range(0,count):   
    os.chdir(changefile)
    fitshdu = fits.open(oripath+filetemp[i])
    data = fitshdu[0].data
    fitsdata = np.copy(data)
    sources1,positions1,mylist =  findsource(fitsdata)
    
    np.savetxt(filetemp[i][:-4]+'.txt', positions1,fmt='%f',delimiter=' ')