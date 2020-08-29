# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 15:22:02 2020

@author: dingxu
"""

import pandas as pd
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from photutils import CircularAperture
from itertools import combinations
import math
import itertools
from time import time
import cv2


#20190603132720Auto.fit
file  = '0.fits'
path = 'E:\\shunbianyuan\\dataxingtuan\\alngc7142\\'
filename = path+file
fitshdu = fits.open(filename)
data = fitshdu[0].data
fitsdata = np.copy(data)
print(fitshdu.info())

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
    
    
    
filestar = 'E:\\shunbianyuan\\phometry\\'+'ngc7142variable.txt'
fileposition = np.loadtxt(filestar)

i = 0
displayimage(fitsdata,1,2)
plt.plot(fileposition[i][0], fileposition[i][1], '*')


xynumpy = np.zeros((4,2))
radecnumpy = np.zeros((4,2))

xynumpy[0][0] = 508.312
xynumpy[0][1] = 3451.329
radecnumpy[0][0] = 327.73664167
radecnumpy[0][1] = 65.26526111

xynumpy[1][0] = 2776.161
xynumpy[1][1] = 3236.242
radecnumpy[1][0] = 325.63530417
radecnumpy[1][1] = 65.30859444

xynumpy[2][0] = 565.788
xynumpy[2][1] = 1049.460
radecnumpy[2][0] = 327.61771667 
radecnumpy[2][1] = 66.19171389

xynumpy[3][0] = 2801.946
xynumpy[3][1] = 995.963
radecnumpy[3][0] = 325.47870833 
radecnumpy[3][1] = 66.17168333
    


src_pts = xynumpy
dst_pts = radecnumpy
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
tempmatrix = np.zeros((3,1),dtype = np.float64)
tempmatrix[2] = 1
tempmatrix[0] = 2633.28     
tempmatrix[1] = 511.132
result = np.dot(H,tempmatrix)
    
ra = result[0]/result[2]
dec = result[1]/result[2]

print(ra, dec)

displayimage(fitsdata,1,1)
plt.plot(tempmatrix[0], tempmatrix[1], '*')





