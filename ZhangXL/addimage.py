# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 11:38:48 2021

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
import imageio

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

gif_images = []
def displayimage(img, coff, i):
    minimg,maximg = adjustimage(img, coff)
    plt.figure(i)
    plt.imshow(img, cmap='gray', vmin = minimg, vmax = maximg)
    plt.savefig('img.jpg')
    gif_images.append(imageio.imread('img.jpg'))

wrpath = 'E:\\shunbianyuan\\phometry\\pipelinecode\\pipeline\\ZhangXL\\xdata\\'
def witefits(data,name, head):
    writepath = wrpath
    os.chdir(writepath)
    if(os.path.exists(writepath+name + '.fit')):
        os.remove(name + '.fit')

    fitsdata = np.float32(data)
    fits.writeto(name + '.fit', fitsdata, head)

filetemp = []
count = 0
oripath = 'E:\\shunbianyuan\\phometry\\pipelinecode\\pipeline\\ZhangXL\\xdata\\'  #路径参数
for root, dirs, files in os.walk(oripath):
   for file in files:
       if (file[-4:] == '.fit'):
           count = count+1
           filetemp.append(file)
    
tempdata = []      
for i in range(0, count):
    fitshdu = fits.open(oripath+filetemp[i])
    data = fitshdu[0].data  
    fitsdata = np.copy(data)
    tempdata.append(fitsdata)
    
headdata = fitshdu[0].header
data =  tempdata[0]+tempdata[1]+tempdata[2]+tempdata[3]+tempdata[4]+tempdata[5]  

witefits(data, 'coo', headdata)   