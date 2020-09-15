# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 10:55:29 2020

@author: dingxu
"""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import shutil

filetemp = []
count = 0
oripath = 'H:\\wuzy\\'  #路径参数
for root, dirs, files in os.walk(oripath):
   for file in files:
       if (file[-5:] == '.fits'):
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
    



file10 = 'H:\\wuzy\\10\\'
file60 = 'H:\\wuzy\\60\\'


for i in range(count):      
    if (int(filetemp[i][:-1])/2 == 0):
        shutil.copy(oripath+filetemp[i], file10)
        
    if (int(filetemp[i][:-1])/2 == 1):
        shutil.copy(oripath+filetemp[i], file60)
        
   

    #displayimage(imgdata,1,0)
    #plt.pause(0.01)
    #plt.clf()
