# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 22:10:03 2020

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

i = 2
j = 4
fitsdata = np.copy(data[796*i:796+796*i,778*j:778+778*j])
print(i,j)

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
    for i in range(hang):
        delt = np.sqrt((targetx - sumpho[i][0])**2+(targety - sumpho[i][1])**2)
        if delt < threshold:
            return sumpho[i]
   
displayimage(data, 1, 0) 
displayimage(fitsdata, 1, 1) 


datatime = np.loadtxt('datatime.txt')
starlight = np.loadtxt('starlight.txt')

x1 = 595
y1 = 88
xyflux1 = findtarget(778*j+x1, 796*i+y1, starlight)
plt.plot(xyflux1[0]-778*j,xyflux1[1]-796*i,'*')

x2 = 578
y2 = 142
xyflux2 = findtarget(778*j+x2, 796*i+y2, starlight)
plt.plot(xyflux2[0]-778*j,xyflux2[1]-796*i,'*')


plt.figure(2)
plt.plot(datatime, xyflux2[2:]-xyflux1[2:],'.')

print(np.std(xyflux2[2:]-xyflux1[2:]))

'''
tempflux = np.copy(starlight)
hang,lie = tempflux.shape
for m in range(hang):
    if tempflux[m,0] >= 778*j and tempflux[m,0] <= 778+778*j and tempflux[m,1] >= 796*i and tempflux[m,1] <= 796+796*i:
        tempflux[m,2:] = tempflux[m,2:]-xyflux1[2:]
        plt.figure(4)
        plt.title(str(m))
        plt.plot(datatime,  tempflux[m,2:], '.')
        plt.pause(1)
        plt.clf()
        displayimage(fitsdata, 1, 5)
        plt.plot(tempflux[m,0]-778*j,tempflux[m,1]-796*i,'*')
        plt.pause(0.1)
        plt.clf()
        
#np.savetxt('tempflux'+str(i)+str(j)+'.txt', tempflux)
'''  
