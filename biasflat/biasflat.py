# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 16:38:42 2020

@author: dingxu
"""

import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from photutils import CircularAperture, CircularAnnulus
from photutils import aperture_photometry
from astropy.time import Time




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


imgfile = 'E:\\shunbianyuan\\dataxingtuan\\ngc7142cmd\\'+'YFDh050234.fits' #目标图像
imgdata = readdata(imgfile,1)
displayimage(imgdata, 3 ,1)

biasfile = 'E:\\shunbianyuan\\dataxingtuan\\ngc7142cmd\\biasimgbin\\'+'bias.fit'
biasdata = readdata(biasfile,0)
displayimage(biasdata, 3 ,2)

flatfile = 'E:\\shunbianyuan\\dataxingtuan\\ngc7142cmd\\flatbinv\\'+'flatv.fit' #目标滤光片
flatdata = readdata(flatfile,0)
displayimage(flatdata, 3 ,3)


Biasmean = np.mean(biasdata)
flat = flatdata-Biasmean
flatmean = np.mean(flat)


guiyi = (imgdata-Biasmean)/flat
gaizheng = flatmean*guiyi+Biasmean
gaizheng = gaizheng[102:980,102:980]
displayimage(gaizheng, 5 ,4)

path = 'E:\\shunbianyuan\\dataxingtuan\\ngc7142cmd\\'
def witefits(data,name):
    data = np.float32(data)
    filename = path + name + '.fit'
    print(filename)
    if os.path.exists(filename):
        os.remove(filename)
    grey=fits.PrimaryHDU(data)
    greyHDU=fits.HDUList([grey])
    greyHDU.writeto(path+name + '.fit')
    
witefits(gaizheng,'imgv')   
