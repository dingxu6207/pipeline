# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:49:37 2020

@author: dingxu
"""
import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import astroalign as aa
from photutils import CircularAperture, CircularAnnulus
from photutils import aperture_photometry
from astropy.time import Time

def readdata(filename, i=0):
    fitshdu = fits.open(filename)
    data = fitshdu[i].data   
    fitsdata = np.copy(data)
    return fitsdata

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
    
path = 'E:\\shunbianyuan\\dataxingtuan\\ngc7142cmd\\'
def witefits(data,name):
    data = np.float32(data)
    filename = path + name + '.fit'
    print(filename)
    if os.path.exists(filename):
        os.remove(filename)
    grey=fits.PrimaryHDU(data)
    greyHDU=fits.HDUList([grey])
    greyHDU.writeto(path+name + '.fit')
    
imgb = 'E:\\shunbianyuan\\dataxingtuan\\ngc7142cmd\\'+'imgb.fit'
imgbdata = readdata(imgb)
displayimage(imgbdata, 3 , 1)

imgv = 'E:\\shunbianyuan\\dataxingtuan\\ngc7142cmd\\'+'imgv.fit'
imgvdata = readdata(imgv)
displayimage(imgvdata, 3 , 2)

aligned_image, footprint = aa.register(imgbdata, imgvdata)
displayimage(aligned_image, 3 , 3)
witefits(aligned_image, 'newimgbdata')
