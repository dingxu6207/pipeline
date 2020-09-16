# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 16:37:20 2020

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

def displayimage(img1, coff, i):
    img = np.copy(img1)
    minimg,maximg = adjustimage(img, coff)
    plt.figure(i)
    plt.imshow(img, cmap='gray', vmin = minimg, vmax = maximg)
    #plt.savefig(str(i)+'.jpg')


def findsource(img1):  
    img = np.copy(img1)
    mean, median, std = sigma_clipped_stats(img, sigma=3.0) 
    daofind = DAOStarFinder(fwhm = 2.3, threshold=5.*std)
    sources = daofind(img - median)

    #tezhen = np.transpose((sources['sharpness'], sources['roundness1'],sources['flux']))
    #print(sources[0])
    tezhen = np.transpose((sources['xcentroid'], sources['ycentroid']))
    posiandmag = np.transpose((sources['xcentroid'], sources['ycentroid'],sources['flux']))

    return tezhen,posiandmag.tolist()

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

def aliigendata(data11, data22, lenstar = 30):
    #找星
    data1 = np.copy(data11)
    data2 = np.copy(data22)
    positions1,posiandmag1 =  findsource(data1)
    positions2,posiandmag2 =  findsource(data2)
       
    #进行排序    
    posiandmag1.sort(key=lambda x:x[2],reverse=True)
    posiandmag2.sort(key=lambda x:x[2],reverse=True)
    
    #选前面亮星
    #lenstar = 25
    posiandmag1 = posiandmag1[0:lenstar]
    posiandmag2 = posiandmag2[0:lenstar]
    
    #构成三角形
    sanjiao1 = list(combinations(posiandmag1,3))
    sanjiao2 = list(combinations(posiandmag2,3))
    
    #算距离		
    lensan1 = len(sanjiao1)
    temp1 = [julisanjiao(sanjiao1,i) for i in range (0,lensan1)]
    
    lensan2 = len(sanjiao2)
    temp2 = [julisanjiao(sanjiao2,i) for i in range (0,lensan2)] 
		
    #匹配
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
        
        pan1 = abs(oneju0-twoju0)
        pan2 = abs(oneju1-twoju1)
        pan3 = abs(oneju2-twoju2)
        
        if (pan1 < 0.1)and(pan2<0.1)and(pan3<0.1):
            pitemp1.append(i[0])
            pitemp2.append(i[1])
            count = count+1    
	 
    #几何变换
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
      
    return src_pts,dst_pts

 
def newdata(datainput, dataref):
    hang1,lie1 = datainput.shape
    datainput = np.float32(datainput)
    dataref = np.float32(dataref)
    src_pts,dst_pts =  aliigendata(datainput, dataref)    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    newdata = cv2.warpPerspective(datainput, H, (lie1,hang1))
    return newdata
