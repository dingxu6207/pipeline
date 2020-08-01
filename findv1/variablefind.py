# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 09:34:48 2020

@author: dingxu
"""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt


file  = '0.fits'
path = 'E:\\shunbianyuan\\dataxingtuan\\alngc7142\\'
filename = path+file
fitshdu = fits.open(filename)
data = fitshdu[0].data

i = 4
j = 4
#hang = 0
fitsdata = np.copy(data[796*i:796+796*i,778*j:778+778*j])

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
    
def findtarget(targetx, targety, sumpho, threshold=10):
    hang,lie = sumpho.shape    
    for m in range(hang):
        delt = np.sqrt((targetx - sumpho[m][0])**2+(targety - sumpho[m][1])**2)
        if delt < threshold:
            print('cvm=', m)
            return sumpho[m]
   
displayimage(data, 1, 0) 
displayimage(fitsdata, 1, 1) 

datatime = np.loadtxt('datatime.txt')
starlight = np.loadtxt('starlight.txt')


x1 = 118
y1 = 349
xyflux1 = findtarget(778*j+x1, 796*i+y1, starlight)
plt.plot(xyflux1[0]-778*j,xyflux1[1]-796*i,'*')

x2 = 327
y2 = 533
xyflux2 = findtarget(778*j+x2, 796*i+y2, starlight)
plt.plot(xyflux2[0]-778*j,xyflux2[1]-796*i,'*')


plt.figure(2)
plt.plot(datatime, xyflux2[2:]-xyflux1[2:],'.')

tempflux = np.copy(starlight)
hang,lie = tempflux.shape
magflux = np.copy(starlight)

jiaochatemp= []
sigmatemp = []
for m in range(hang):
    if tempflux[m,0] >= 778*j and tempflux[m,0] <= 778+778*j and tempflux[m,1] >= 796*i and tempflux[m,1] <= 796+796*i:
        sigmatemp.append(magflux[m,2:])
        tempflux[m,2:] = tempflux[m,2:]-xyflux1[2:]
        #tempflux[m,2:] = tempflux[m,2:]
        jiaochatemp.append(tempflux[m])
        
        displayimage(fitsdata, 1, 3)
        plt.plot(tempflux[m,0]-778*j,tempflux[m,1]-796*i,'*')
        plt.pause(0.1)
        plt.clf()
        
        plt.figure(4)
        plt.title(str(m))
        plt.plot(datatime,  tempflux[m,2:], '.')
        plt.pause(0.05)
        plt.clf()
        
arrayjiaocha = np.array(jiaochatemp)
np.savetxt('arrayjiaocha.txt', arrayjiaocha)

arraysigma = np.array(sigmatemp)
np.savetxt('sigma.txt', arraysigma)