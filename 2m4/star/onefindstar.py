# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 14:00:24 2020

@author: dingxu
"""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt



x,y = 336,394

file  = 'YFCa260278.fit'
path = 'E:\\shunbianyuan\\Asteroids_Dingxu\\6478\\20190126_6478\\alligendata\\'

filename = path+file
fitshdu = fits.open(filename)
data = fitshdu[0].data

fitsdata = np.copy(data)

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
    
displayimage(fitsdata, 1, 0) 
plt.plot(x,y,'*')

