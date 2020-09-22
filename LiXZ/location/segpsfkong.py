# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 17:01:21 2020

@author: dingxu
"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from photutils import CircularAperture
from scipy.optimize import curve_fit
from scipy import asarray as ar

ib = 4 #行扫描 i = 21
jb = 1#列扫描 j=20
print(ib,jb)
position = np.loadtxt('location.txt')
hang,lie = position.shape

tempposition = []
position1 = []
for i in range(hang):
    tempposition.append(tuple(position[i]))
    for j in range(i+1,hang):
        temp1 = position[i]
        temp2 = position[j]
        d1 = (temp1[0]-temp2[0])**2
        d2 = (temp1[1]-temp2[1])**2
        distance = np.sqrt((d1+d2))
        
        if (distance < 17 ):
            position1.append(tuple(temp1))
            position1.append(tuple(temp2))
        
        

list1 = list(set(position1))
positionarr = np.array(list1)

cha = list(set(tempposition)^set(list1))
arraycha = np.array(cha)

#20190603132720Auto.fit
#file  = 'imgv.fit'
#path = 'E:\\shunbianyuan\\dataxingtuan\\ngc7142cmd\\'
file  = 'd4738787L018m000.fit'
path = 'E:\\shunbianyuan\\dataxingtuan\\berkeley99\\'
filename = path+file
fitshdu = fits.open(filename)
data = fitshdu[0].data
#ib = 0 #行扫描 i = 21
#jb = 0#列扫描 j=20
fitsdata = data[796*ib:796+796*ib,778*jb:778+778*jb]
#796*i:796+796*i,778*j:778+778*j
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
    
apertures1 = CircularAperture(positionarr, r=6.)   
displayimage(fitsdata, 1 , 1)
apertures1.plot(color='blue', lw=1.5, alpha=0.5)


apertures2 = CircularAperture(arraycha, r=7.)   
displayimage(fitsdata, 1 , 2)
apertures2.plot(color='blue', lw=1.5, alpha=0.5)

apertures3 = CircularAperture(position, r=7.)   
displayimage(fitsdata, 1 , 3)
apertures3.plot(color='blue', lw=1.5, alpha=0.5)

np.savetxt('locationsan'+str(ib)+str(jb)+'.txt', arraycha)
np.savetxt('locationmi'+str(ib)+str(jb)+'.txt', positionarr)
