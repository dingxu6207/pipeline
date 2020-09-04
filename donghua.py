# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 13:35:34 2020

@author: dingxu
"""
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os


filetemp = []
count = 0
oripath = 'E:\\shunbianyuan\\dataxingtuan\\ngc7142\\'  #路径参数
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


def displayimage(img, coff, i):
    minimg,maximg = adjustimage(img, coff)
    plt.figure(i)
    plt.imshow(img, cmap='gray', vmin = minimg, vmax = maximg)
    
    
 
for i in range(count):
    fitshdu = fits.open(oripath+filetemp[i])
    imgdata = fitshdu[0].data
    displayimage(imgdata,1,0)
    plt.pause(0.01)
    plt.clf()
    