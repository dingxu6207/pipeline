# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 23:08:58 2020

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
import trifunc
#import sajiaofunc

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
oripath = 'E:\\shunbianyuan\\dataxingtuan\\berkeley99\\'  #路径参数
for root, dirs, files in os.walk(oripath):
   for file in files:
       if (file[-4:] == '.fit'):
           count = count+1
           filetemp.append(file)
 
print(count)   
wrpath = 'E:\\shunbianyuan\\dataxingtuan\\alberkeley99\\'
def witefits(data,name, time):
    writepath = wrpath
    os.chdir(writepath)
    if(os.path.exists(writepath+name + '.fit')):
        os.remove(name + '.fit')
    
    hdr = fits.Header()
    hdr['DATE'] = time
    fitsdata = np.float32(data)
    fits.writeto(name + '.fit', fitsdata, hdr)
    

zampledata = readdata(oripath+filetemp[0], 0) 
print(zampledata.shape)
#displayimage(zampledata, 1 ,4)  
       


for i in range(0, count):
    fitshdu = fits.open(oripath+filetemp[i])
    data = fitshdu[0].data   
    fitsdata = np.copy(data)
        
    headdata = fitshdu[0].header
    DATEOBS = headdata['DATE-OBS']
    TIME = headdata['TIME']
    datatime = '20'+DATEOBS[-2:]+'-'+DATEOBS[-5:-3]+'-'+DATEOBS[-8:-6]+'T'+TIME[-10:]
    
    try:
        aligned_image = trifunc.newdata(fitsdata, zampledata) 
        headdata = fitshdu[0].header    
        witefits(aligned_image, filetemp[i][:-4], datatime)   
        #displayimage(aligned_image,1,0)
        #plt.pause(0.1)
        #plt.clf()
        print('it is ok!\n')  
    except:
        print('it is eror!\n') 
        #print(filetemp[i])
        