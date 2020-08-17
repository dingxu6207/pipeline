# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 16:40:01 2020

@author: dingxu
"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from photutils import CircularAperture
import cv2
import os
import math
from itertools import combinations,permutations
from time import time
import ois
import itertools

#半高全宽和匹配数目修改即可
fitsname1 = 'E:\\shunbianyuan\\newdata\\'+'M31.fts'
fitsname2 = 'E:\\shunbianyuan\\newdata\\'+'M31o.fts'
onehdu = fits.open(fitsname1)
imgdata1 = onehdu[0].data  #hdu[0].header

copydata1 = np.copy(imgdata1)
imgdata1 = np.float32(copydata1)
imgdata1 = np.rot90(imgdata1)
imgdata1 = np.rot90(imgdata1)
oneimgdata = imgdata1
hang1,lie1 = oneimgdata.shape

twohdu = fits.open(fitsname2)
imgdata2 = twohdu[0].data  #hdu[0].header
#imgdata2 = np.rot90(imgdata2)
copydata2 = np.copy(imgdata2)
imgdata2 = np.float32(copydata2)
twoimgdata = imgdata2
hang2,lie2 = twoimgdata.shape


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
    #plt.savefig(str(i)+'.jpg')


def findsource(img):    
    mean, median, std = sigma_clipped_stats(img, sigma=3.0) 
    daofind = DAOStarFinder(fwhm = 4, threshold=5.*std)
    sources = daofind(img - median)

    #tezhen = np.transpose((sources['sharpness'], sources['roundness1'],sources['flux']))
    #print(sources[0])
    tezhen = np.transpose((sources['xcentroid'], sources['ycentroid']))
    posiandmag = np.transpose((sources['xcentroid'], sources['ycentroid'],sources['flux']))

    return tezhen,posiandmag.tolist()


###实现找星###
positions1,posiandmag1 =  findsource(oneimgdata)
positions2,posiandmag2 =  findsource(twoimgdata)

lenstar1 = len(positions1)
lenstar2 = len(positions2)


apertures1 = CircularAperture(positions1, r=13.)
apertures2 = CircularAperture(positions2, r=13.)




start = time()
print("Start: " + str(start))

posiandmag1.sort(key=lambda x:x[2],reverse=True)
posiandmag2.sort(key=lambda x:x[2],reverse=True)

##选19颗亮星
#lenstar = min(lenstar1,lenstar2)
lenstar = 30
index = 2
posiandmag1 = posiandmag1[lenstar*index:lenstar+lenstar*index]
posiandmag2 = posiandmag2[lenstar*index:lenstar+lenstar*index]

sanjiao1 = list(combinations(posiandmag1,3))
sanjiao2 = list(combinations(posiandmag2,3))

def julisanjiao(sanjiao1,i):
    x1 = sanjiao1[i][0][0]
    y1 = sanjiao1[i][0][1]
    
    x2 = sanjiao1[i][1][0]
    y2 = sanjiao1[i][1][1]
    
    x3 = sanjiao1[i][2][0]
    y3 = sanjiao1[i][2][1]
    
    datadis1 = ((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
    dS1S2 = math.sqrt(datadis1)
    
    datadis2 = ((x1-x3)*(x1-x3)+(y1-y3)*(y1-y3))
    dS1S3 = math.sqrt(datadis2)
    
    datadis3 = ((x2-x3)*(x2-x3)+(y2-y3)*(y2-y3))
    dS2S3 = math.sqrt(datadis3)
       
    return [[x1,y1],[x2,y2],[x3,y3],[dS1S2,dS1S3,dS2S3]]

lensan1 = len(sanjiao1)
temp1 = [julisanjiao(sanjiao1,i) for i in range (0,lensan1)]
#temp1 = []
#for i in range (0,lensan1):
#    jie1 = julisanjiao(sanjiao1,i)
#    temp1.append(jie1)
    
lensan2 = len(sanjiao2)
temp2 = [julisanjiao(sanjiao2,i) for i in range (0,lensan2)]    
#temp2 = []
#for i in range (0,lensan2):
#    jie2 = julisanjiao(sanjiao2,i)
#    temp2.append(jie2)

pitemp1 = []
pitemp2 = []   
count = 0 

for i in itertools.product(temp1, temp2):
    oneju0 = i[0][3][0]
    oneju1 = i[0][3][1]
    oneju2 = i[0][3][2]
    oneab = oneju0/oneju1
    onebc = oneju1/oneju2
    oneca = oneju2/oneju0
        
    twoju0 = i[1][3][0]
    twoju1 = i[1][3][1]
    twoju2 = i[1][3][2]
    twoab = twoju0/twoju1
    twobc = twoju1/twoju2
    twoca = twoju2/twoju0
        
    pan1 = abs(oneab-twoab)
    pan2 = abs(onebc-twobc)
    pan3 = abs(oneca-twoca)
        
    if (pan1 < 0.0001)and(pan2<0.0001)and(pan3<0.0001):
        pitemp1.append(i[0])
        pitemp2.append(i[1])
        count = count+1    

        
stop = time()
print("Stop: " + str(stop))
print(str(stop-start) + "秒") 


            
displayimage(oneimgdata,3,0)
apertures1.plot(color='blue', lw=1.5, alpha=0.5)
#plt.plot(pitemp1[0][0][0],pitemp1[0][0][1],'*')

displayimage(twoimgdata,3,1)
apertures2.plot(color='blue', lw=1.5, alpha=0.5)
#plt.plot(pitemp2[0][0][0],pitemp2[0][0][1],'*')    

hmerge = np.hstack((oneimgdata, twoimgdata)) #水平拼接
displayimage(hmerge, 3, 2) 
   

srckp1 = []
srckp2 = []
for i in range(0,count):
    for j in range(0,3):
        x10 = pitemp1[i][j][0]
        x11 = pitemp2[i][j][0]
            
        y10 = pitemp1[i][j][1]
        y11 = pitemp2[i][j][1]
        
        srckp1.append(x10)
        srckp1.append(y10)
        srckp2.append(x11)
        srckp2.append(y11)
        src_pts = np.float32(srckp1).reshape(-1,2)
        dst_pts = np.float32(srckp2).reshape(-1,2)
    
        lie1 = imgdata1.shape[1]
        plt.plot([x10,x11+lie1],[y10,y11],linewidth = 0.8)  


H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)    
newimg = cv2.warpPerspective(oneimgdata, H, (lie1,hang1))

displayimage(newimg, 1, 3) 
minusimg = np.float32(newimg) - np.float32(imgdata2)  
displayimage(minusimg, 3, 4) 
displayimage(newimg, 1, 5)  
print(H)

