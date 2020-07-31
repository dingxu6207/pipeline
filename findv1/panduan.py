# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 21:53:34 2020

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


file  = '0.fits'
path = 'E:\\shunbianyuan\\dataxingtuan\\alngc7142\\'
filename = path+file
fitshdu = fits.open(filename)
imgdata = fitshdu[0].data

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





time = np.loadtxt('datatime.txt')
data = np.loadtxt('arrayjiaocha.txt')


a = 5.342944332436061e-08
b = 0.8114966574650161

def funcexp(xdata):
    y = a*np.exp(b*xdata)
    return y



hang,lie = data.shape

romstemp = []
for i in range(hang):
    midium = np.median(data[i,2:])
    sigma = np.log10(funcexp(data[i,2:]))
    wjz = np.abs((data[i,2:] - midium)/sigma)
    Roms = np.sum(wjz)
    romstemp.append(Roms)
    
index = 242
plt.figure(0)    
plt.plot(time,data[index, 2:],'.')

displayimage(imgdata, 1 ,1)
plt.plot(data[index,0], data[index,1], '*')