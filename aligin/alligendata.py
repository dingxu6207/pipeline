# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 05:15:22 2020

@author: dingxu
"""

import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from photutils import CircularAperture, CircularAnnulus
from photutils import aperture_photometry
from astropy.time import Time
import astroalign as aa
import cv2


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

filename1 = 'E:\\shunbianyuan\\dataxingtuan\\berkeley99\\'+'d4738787L018m000.fit'
filename2 = 'E:\\shunbianyuan\\dataxingtuan\\berkeley99\\'+'d4738787L018m001.fit'

imgdata1 = readdata(filename1, 0)
imgdata2 = readdata(filename2, 0)

displayimage(imgdata1, 1 , 0)
displayimage(imgdata2, 1 , 1)


transf, (s_list, t_list) = aa.find_transform(imgdata1, imgdata2)
H, mask = cv2.findHomography(s_list, t_list, cv2.RANSAC,5.0)


hmerge = np.hstack((imgdata1, imgdata2)) #水平拼接
displayimage(hmerge, 1, 2) 

hangcount = len(s_list)
hang,lie = imgdata1.shape
for i in range (0,hangcount):
    
    x10 = s_list[i,0]
    y10 = s_list[i,1]
    
    x11 = t_list[i,0]
    y11 = t_list[i,1]
    plt.plot([x10,x11+lie],[y10,y11],linewidth = 0.8)