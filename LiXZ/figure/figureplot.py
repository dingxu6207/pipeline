# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 23:01:39 2020

@author: dingxu
"""

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from photutils import CircularAperture
from PIL import ImageDraw
from PIL import ImageFont

filename1 = 'E:\\shunbianyuan\\dataxingtuan\\alberkeley99\\'+'d4738787L018m000.fit'
#fitsname2 = 'E:\\BOOTES4\\20181118\\03095\\'+'20181118125001-285-RA.fits'


onehdu = fits.open(filename1)
imgdata1 = onehdu[0].data  #hdu[0].header

#twohdu = fits.open(fitsname2)
#imgdata2 = twohdu[0].data  #hdu[0].header

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
    plt.savefig(str(i)+'.jpg')

position = [(142.581222, 515.658569),(1598.27,746.609), (2131.493345,357.152787),(2491.999756, 725.619976)
           ,(2711.463913, 433.11385),(1816.163652,1245.896673),(2786.189961,956.7287670000001),(3608.404395,831.514009)
           ,(431.000548, 2116.343129),(934.298323,2724.521225),(3721.009938, 2588.202828),(326.870959, 2938.956326)
           ,(830.073831, 3702.420595),(3006.135694, 2910.8089840000002),(1808.134938, 3220.980439),(1458.313908, 3463.575996)
           ]
position = np.array(position)
displayimage(imgdata1,0.3,0)
#displayimage(imgdata2,3,1)
apertures1 = CircularAperture(position, r=7.)
apertures1.plot(color='blue', lw=2.5, alpha=0.5)

titletemp = ['V1', 'V2']
lenposition = len(position)
for i in range(lenposition):  
    plt.text(position[i][0], position[i][1], 'V'+str(i+1), fontsize=10, color = "r", style = "italic", weight = "light", verticalalignment='center', horizontalalignment='left',rotation=0) #给散点加标签

