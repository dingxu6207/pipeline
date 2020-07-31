# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 21:39:01 2020

@author: dingxu
"""

import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from photutils import CircularAperture, CircularAnnulus
from photutils import aperture_photometry
import imageio

file  = '0.fits'
path = 'E:\\shunbianyuan\\dataxingtuan\\alngc7142\\'
filename = path+file
fitshdu = fits.open(filename)
data = fitshdu[0].data
i = 7 #行扫描 i = 21
j = 9#列扫描 j=20
#fitsdata = data[398*i:398+398*i,389*j:389+389*j]
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
    plt.clf()
    plt.imshow(img, cmap='gray', vmin = minimg, vmax = maximg)
    
    

starlight = np.loadtxt('starlight.txt')
hang,lie = starlight.shape

targetx1 = 92
targety1 = 73

targetx2 = 3364
targety2 = 491

targetx3 = 234
targety3 = 3538


targetx4 = 3564
targety4 = 3721


threshold = 10
for i in range(hang):
    delt1 = np.sqrt((targetx1 - starlight[i][0])**2+(targety1 - starlight[i][1])**2)
    delt2 = np.sqrt((targetx2 - starlight[i][0])**2+(targety2 - starlight[i][1])**2)
    delt3 = np.sqrt((targetx3 - starlight[i][0])**2+(targety3 - starlight[i][1])**2)
    delt4 = np.sqrt((targetx4 - starlight[i][0])**2+(targety4 - starlight[i][1])**2)
    if delt1 < threshold:
        print(i)
        m1 = i
    if delt2 < threshold:
        print(i)
        m2 = i
    if delt3 < threshold:
        print(i)
        m3 = i
    if delt4 < threshold:
        print(i)
        m4 = i
          
x = np.arange(0,lie-2,1,dtype=np.int16)

for i in range(1246, hang):
    
    displayimage(fitsdata, 1, 0)   
    plt.plot(starlight[i,0], starlight[i,1],'*')
    plt.plot(targetx1, targety1,'*')
    plt.plot(targetx2, targety2,'*')
    plt.plot(targetx3, targety3,'*')
    plt.plot(targetx4, targety4,'*')
    plt.pause(1)
      
    plt.figure(1)
    plt.clf()        
    fluxy = starlight[i,2:]-(starlight[m1,2:]+starlight[m2,2:]+starlight[m3,2:]+starlight[m4,2:])/4
    plt.plot(x,fluxy,'.')
    
    nihe = np.polyfit(x, fluxy, 9)
    nihey = np.poly1d(nihe)
    y = nihey(x)
    plt.plot(x, y, label="xx")
    
    plt.hlines(np.mean(fluxy)+2*np.std(fluxy), 0, lie,color="red")#横线
    plt.hlines(np.mean(fluxy)-2*np.std(fluxy), 0, lie,color="red")#横线
   
    plt.legend()
    plt.title(str(i))
    plt.pause(2)
    

