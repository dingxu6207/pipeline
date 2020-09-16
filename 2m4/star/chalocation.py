# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:15:05 2020

@author: dingxu
"""

import numpy as np
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

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
    
oneimg = 'E:\\shunbianyuan\\Asteroids_Dingxu\\6478\\20200826_6478\\alligen\\YFDh260160.fit' 
twoimg = 'E:\\shunbianyuan\\Asteroids_Dingxu\\6478\\20200826_6478\\alligen\\YFDh260310.fit' 

onedata = readdata(oneimg, 0)
twodata = readdata(twoimg, 0)

displayimage(onedata, 1 ,0)
plt.plot(373.903924, 400.986487, '*')
displayimage(twodata, 1 ,1)
plt.plot(308.568518, 241.628884, '*')

location = 'E:\\shunbianyuan\\Asteroids_Dingxu\\6478\\20200826_6478\\location\\'

location0 = np.loadtxt(location+'YFDh260160.txt')

location1 = np.loadtxt(location+'YFDh260310.txt')


def sourcephotometry(targetx, targety, sumpho, threshold=3):
    hang,lie = sumpho.shape    
    for i in range(hang):
        delt = np.sqrt((targetx - sumpho[i][0])**2+(targety - sumpho[i][1])**2)
        if delt < threshold:
            print(sumpho[i])
            

kaishi =  sourcephotometry(374,400,location0)   
zuihou =  sourcephotometry(308,241,location1)         