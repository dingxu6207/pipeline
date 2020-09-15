# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:02:33 2020

@author: dingxu
"""

import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

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
oripath = 'E:\\shunbianyuan\\Asteroids_Dingxu\\6478\\20200825_6478\\flat\\'  #路径参数
for root, dirs, files in os.walk(oripath):
   for file in files:
       if (file[-5:] == '.fits'):
           count = count+1
           filetemp.append(file)
         
            
count = len(filetemp)   
temp = []        
for i in range(0, count):
    fitshdu = fits.open(oripath+filetemp[i])
    data = fitshdu[1].data   
    fitsdata = np.copy(data)
    print(fitsdata.shape)
    temp.append(fitsdata)
    headdata = fitshdu[0].header
    
sum = 0    
for i in range(0,count):
    sum = sum + temp[i]
    
average = sum/count


displayimage(average,3,1)


def witefits(data,name, head):
    writepath = oripath
    os.chdir(writepath)
    if(os.path.exists(writepath+name + '.fit')):
        os.remove(name + '.fit')

    fitsdata = np.float32(data)
    fits.writeto(name + '.fit', fitsdata, head)
    
witefits(average, 'flat', headdata)   
