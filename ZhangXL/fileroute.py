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
oripath = 'E:\\shunbianyuan\\Asteroids_Dingxu\\228P_phot\\20200215_228P\\'  #路径参数
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
    



dstflat = 'E:\\shunbianyuan\\Asteroids_Dingxu\\228P_phot\\20200215_228P\\flat\\'
dstbias = 'E:\\shunbianyuan\\Asteroids_Dingxu\\228P_phot\\20200215_228P\\bias\\'
dsttarget = 'E:\\shunbianyuan\\Asteroids_Dingxu\\228P_phot\\20200215_228P\\target\\'

for i in range(count):
    fitshdu = fits.open(oripath+filetemp[i])
    imgdata = fitshdu[1].data
    objectname = fitshdu[0].header['OBJECT']
       
    if (objectname == 'Flat_bin_JR'):
        shutil.copy(oripath+filetemp[i], dstflat)
        
    if (objectname == 'Bias_img_bin'):
        shutil.copy(oripath+filetemp[i], dstbias)
        
    if (objectname == '133p_JI'):
        shutil.copy(oripath+filetemp[i], dsttarget)

    #displayimage(imgdata,1,0)
    #plt.pause(0.01)
    #plt.clf()
