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
file  = 'd4738787L018m000.fit'
path = 'E:\\shunbianyuan\\dataxingtuan\\alberkeley99\\'

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
    
    
'''    
filestar = 'E:\\shunbianyuan\\phometry\\'+'ngc7142variable.txt'
fileposition = np.loadtxt(filestar)

i = 0
displayimage(fitsdata,1,2)
plt.plot(fileposition[i][0], fileposition[i][1], '*')
'''

xynumpy = np.zeros((4,2))
radecnumpy = np.zeros((4,2))

xynumpy[0][0] = 504.011
xynumpy[0][1] = 760.424
radecnumpy[0][0] = 352.23455
radecnumpy[0][1] = 72.27539722

xynumpy[1][0] = 516.851
xynumpy[1][1] = 388.087
radecnumpy[1][0] = 352.206825
radecnumpy[1][1] = 72.41904167

xynumpy[2][0] = 1556.824
xynumpy[2][1] = 1312.508
radecnumpy[2][0] = 350.93217083
radecnumpy[2][1] = 72.04836944

xynumpy[3][0] = 2455.567
xynumpy[3][1] = 593.514
radecnumpy[3][0] = 349.74974167
radecnumpy[3][1] = 72.30671944
    


src_pts = xynumpy
dst_pts = radecnumpy
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
tempmatrix = np.zeros((3,1),dtype = np.float64)
tempmatrix[2] = 1
tempmatrix[0] = 1458.313908
tempmatrix[1] = 3463.575996
result = np.dot(H,tempmatrix)
    
ra = result[0]/result[2]
dec = result[1]/result[2]

print(ra, dec)

displayimage(fitsdata,1,1)
plt.plot(tempmatrix[0], tempmatrix[1], '*')





