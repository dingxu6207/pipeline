# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 20:50:54 2020

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

light = np.loadtxt('arrayjiaocha.txt')
datatime = np.loadtxt('datatime.txt')
hang = 12
plt.figure(0)
plt.plot(datatime, light[90,2:], '.')

file  = '0.fits'
path = 'E:\\shunbianyuan\\dataxingtuan\\alngc7142\\'
filename = path+file
fitshdu = fits.open(filename)
data = fitshdu[0].data
imgdata = np.copy(data)
ib = 0
jb = 4
#hang = 0
fitsdata = np.copy(imgdata[796*ib:796+796*ib,778*jb:778+778*jb])

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
    #plt.clf()
    plt.imshow(img, cmap='gray', vmin = minimg, vmax = maximg)
  
displayimage(data, 1, 1)
plt.plot(778*jb+light[90][0], 796*ib+light[90][1], '*')    
print(778*jb+light[90][0], 796*ib+light[90][1])
    
displayimage(fitsdata, 1, 2)
plt.plot(light[90][0], light[90][1], '*')


#plt.plot(2861.421891,470.522248, '*')

