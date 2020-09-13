# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 17:28:33 2020

@author: dingxu
"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from photutils import CircularAperture
from scipy.optimize import curve_fit
from scipy import asarray as ar
import astroalign as aa
import imageio
import ois


#20190603132720Auto.fit
file0 = '1.fits'
file1  = 'L20190406_10053_202408 0411_60S_VR_335680.fits'

path = 'E:\\AST3\\xuyi\\'
filename0 = path+file0
fitshdu0 = fits.open(filename0)
data0 = fitshdu0[0].data
fitsdata0 = np.copy(data0)

filename1 = path+file1
fitshdu1 = fits.open(filename1)
data1 = fitshdu1[0].data
fitsdata1 = np.copy(data1)

m,n = 0,1
segimg0 = fitsdata0[528*m:528*m+528, 528*n:528*n+528]
segimg1 = fitsdata1[528*m:528*m+528, 528*n:528*n+528]

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
    daofind = DAOStarFinder(fwhm = 5, threshold=3.*std)
    sources = daofind(img - median)

    for col in sources.colnames:
        sources[col].info.format = '%.8g'  # for consistent table output
        #print(sources)

    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    positionflux = np.transpose((sources['xcentroid'], sources['ycentroid'],  sources['flux']))
    mylist = positionflux.tolist()
    
    return sources,positions,mylist

def displayimage(img, coff, i):
    minimg,maximg = adjustimage(img, coff)
    plt.figure(i)
    plt.imshow(img, cmap='gray', vmin = minimg, vmax = maximg)
    plt.savefig(str(i)+'.jpg')
    
i = 0
for m in range(0,20):
    for n in range(0,20):
        segimg0 = fitsdata0[528*m:528*m+528, 528*n:528*n+528]
        segimg1 = fitsdata1[528*m:528*m+528, 528*n:528*n+528]
        aligned_image, footprint = aa.register(segimg0, segimg1)
        diff = ois.optimal_system(segimg1, aligned_image)[0]
        displayimage(diff, 3, i)
        i = i+1
        print('it is ok!')


        