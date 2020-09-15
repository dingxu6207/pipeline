# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 16:38:42 2020

@author: dingxu
"""

import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from photutils import CircularAperture, CircularAnnulus
from photutils import aperture_photometry
from astropy.time import Time




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


#imgfile = 'E:\\shunbianyuan\\dataxingtuan\\ngc7142cmd\\'+'YFDh050234.fits' #目标图像
#imgdata = readdata(imgfile,1)
#displayimage(imgdata, 3 ,1)

biasfile = 'E:\\shunbianyuan\\Asteroids_Dingxu\\6478\\20190131_6478\\bias\\'+'bias.fit'
biasdata = readdata(biasfile,0)
displayimage(biasdata, 3 ,2)

flatfile = 'E:\\shunbianyuan\\Asteroids_Dingxu\\6478\\20190131_6478\\flat\\'+'flat.fit' #目标滤光片
flatdata = readdata(flatfile,0)
displayimage(flatdata, 3 ,3)


Biasmean = np.mean(biasdata)
flat = flatdata-Biasmean
flatmean = np.mean(flat)

filetemp = []
count = 0
oripath = 'E:\\shunbianyuan\\Asteroids_Dingxu\\6478\\20190131_6478\\target\\'  #路径参数
for root, dirs, files in os.walk(oripath):
   for file in files:
       if (file[-5:] == '.fits'):
           count = count+1
           filetemp.append(file)
 
wrpath = 'E:\\shunbianyuan\\Asteroids_Dingxu\\6478\\20190131_6478\\probiasflat\\'
def witefits(data,name, head):
    writepath = wrpath
    os.chdir(writepath)
    if(os.path.exists(writepath+name + '.fit')):
        os.remove(name + '.fit')

    fitsdata = np.float32(data)
    fits.writeto(name + '.fit', fitsdata, head)

          
for i in range(0, count):
    fitshdu = fits.open(oripath+filetemp[i])
    data = fitshdu[1].data   
    fitsdata = np.copy(data)
    guiyi = (fitsdata-Biasmean)/flat
    gaizheng = flatmean*guiyi+Biasmean
    gaizheng = gaizheng[150:920,150:920]
    
    headdata = fitshdu[0].header
    
    witefits(gaizheng, filetemp[i][:-5], headdata)   

    displayimage(gaizheng,1,0)
    plt.pause(0.1)
    plt.clf()

