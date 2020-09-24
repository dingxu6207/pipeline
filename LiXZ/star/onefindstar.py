# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 14:00:24 2020

@author: dingxu
"""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

lightdata = np.loadtxt('starlight.txt')

mhang = 2140
x,y = lightdata[mhang,0:2]

x,y = 2131.493345, 357.152787
file  = 'd4738787L018m000.fit'
path = 'E:\\shunbianyuan\\dataxingtuan\\alberkeley99\\'
filename = path+file
fitshdu = fits.open(filename)
data = fitshdu[0].data

j = int(x/389) #行扫描 i = 21
i = int(y/398)#列扫描 j=20

print(i,j)
fitsdata = np.copy(data[398*i:398+398*i,389*j:389+389*j])

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
x1 = x-389*j
y1 = y - i*398
plt.plot(x1,y1,'*')

displayimage(data, 1, 1) 
plt.plot(x,y,'*')