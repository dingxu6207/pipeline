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
magdata = np.loadtxt('sigma.txt')

a = 2.4314987216301876e-07
b = 0.7200849214208086

def funcexp(xdata):
    y = a*np.exp(b*xdata)
    return y



hang,lie = data.shape

romstemp = []
for i in range(hang):
    midium = np.median(data[i,2:])
    sigma = np.log10(funcexp(magdata[i]))
    wjz = np.abs((data[i,2:] - midium)/sigma)
    #wjz = np.abs((data[i,2:] - midium))
    Roms = np.sum(wjz)/268
    #romstemp.append(Roms)
    romstemp.append(np.std(data[i,2:]))
    
index = 129
flagrun = 0
plt.figure(0)    
print(data[index, 0:2])
plt.plot(time,data[index, 2:],'.')


displayimage(imgdata, 1 ,1)
plt.plot(data[index,0], data[index,1], '*')


plt.figure(2)
plt.plot(romstemp, '.')

temp = []
for i in range(len(romstemp)):
    if (romstemp[i]>0.03):#0.07
        temp.append(i)

if flagrun == 1:    
    for i in range(len(temp)):
        t = temp[i]
        plt.figure(3)
        plt.plot(time,data[t, 2:],'.')
        plt.title(str(i))
        plt.pause(1)
        plt.clf()