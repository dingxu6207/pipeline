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
hang = 154
plt.figure(0)
plt.plot(datatime, light[hang,2:], '.')
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
plt.xlabel('JD',fontsize=14)
plt.ylabel('mag',fontsize=14)

file  = '0.fits'
path = 'E:\\shunbianyuan\\dataxingtuan\\alngc7142\\'
filename = path+file
fitshdu = fits.open(filename)
data = fitshdu[0].data
imgdata = np.copy(data)
ib = 4
jb = 4
#hang = 0
fitsdata = np.copy(imgdata[1024*ib:1024+1024*ib,1024*jb:1024+1024*jb])

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
plt.plot(1024*jb+light[hang][0], 1024*ib+light[hang][1], '*')    
print(1024*jb+light[hang][0], 1024*ib+light[hang][1])
    
displayimage(fitsdata, 1, 2)
plt.plot(light[hang][0], light[hang][1], '*')


#plt.plot(2861.421891,470.522248, '*')

