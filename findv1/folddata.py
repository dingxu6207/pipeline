# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

tempflux = np.copy(starlight)

m = 5436
target = 5004
plt.plot(tempflux[m,0]-778*j,tempflux[m,1]-796*i,'*')
plt.plot(tempflux[target,0]-778*j,tempflux[target,1]-796*i,'*')

plt.figure(2)
plt.plot(datatime, tempflux[target,2:]-tempflux[m,2:],'.')
plt.xlabel('JD',fontsize=14)
plt.ylabel('mag',fontsize=14)
ax = plt.gca()
ax.yaxis.set_ticks_position('right') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向

t = 1.3
phasedata = []

for i in range(len(datatime)):
    pd = datatime[i]/t - int(datatime[i]/t)
    phasedata.append(pd)
       
phase = np.array(phasedata) 

plt.figure(3)
plt.plot(phasedata, tempflux[target,2:]-tempflux[m,2:],'.')
ax = plt.gca()
ax.yaxis.set_ticks_position('right') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向