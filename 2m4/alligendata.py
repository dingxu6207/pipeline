# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 16:16:07 2020

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

def readdata(filename, i):
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




filetemp = []
count = 0
oripath = 'E:\\shunbianyuan\\Asteroids_Dingxu\\6478\\20190126_6478\\newdata\\'  #路径参数
for root, dirs, files in os.walk(oripath):
   for file in files:
       if (file[-4:] == '.fit'):
           count = count+1
           filetemp.append(file)
 
    
wrpath = 'E:\\shunbianyuan\\Asteroids_Dingxu\\6478\\20190126_6478\\alligendata\\'
def witefits(data,name, head):
    writepath = wrpath
    os.chdir(writepath)
    if(os.path.exists(writepath+name + '.fit')):
        os.remove(name + '.fit')

    fitsdata = np.float32(data)
    fits.writeto(name + '.fit', fitsdata, head)

zampledata = readdata(oripath+filetemp[0], 0)  
   
       
for i in range(1, count):
    fitshdu = fits.open(oripath+filetemp[i])
    data = fitshdu[0].data   
    fitsdata = np.copy(data)
    aligned_image, footprint = aa.register(fitsdata, zampledata)
    
    headdata = fitshdu[0].header
    
    witefits(aligned_image, filetemp[i][:-4], headdata)   

    displayimage(aligned_image,1,0)
    plt.pause(0.1)
    plt.clf()
