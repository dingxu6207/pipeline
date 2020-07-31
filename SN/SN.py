# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 14:32:56 2020

@author: dingxu
"""

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

from photutils import CircularAperture, CircularAnnulus
from photutils import aperture_photometry

fitsname1 = 'E:\\shunbianyuan\\dataxingtuan\\alngc7142\\'+'0.fits'

onehdu = fits.open(fitsname1)
imgdata1 = onehdu[0].data  #hdu[0].header

position = np.loadtxt('location.txt')

def photometryimg(positions, img, i):
    
    positionslist = positions.tolist()
    
    aperture = CircularAperture(positionslist, r=4) #2*FWHM
    annulus_aperture = CircularAnnulus(positionslist, r_in=6, r_out=8)#4-5*FWHM+2*FWHM
    apers = [aperture, annulus_aperture]
    
    displayimage(img, 1, i) ###画图1
    aperture.plot(color='blue', lw=0.5)
    annulus_aperture.plot(color='red', lw=0.2)
   
    phot_table = aperture_photometry(img, apers)
    bkg_mean = phot_table['aperture_sum_1'] / annulus_aperture.area
    bkg_sum = bkg_mean * aperture.area
    final_sum = phot_table['aperture_sum_0'] - bkg_sum
    #phot_table['residual_aperture_sum'] = final_sum       
    posflux = np.column_stack((positions, final_sum))  
    #return posflux
    magstar = 25 - 2.5*np.log10(abs(final_sum/1))
    return posflux,magstar

def SignalNoise(positions, img):
    positionslist = positions.tolist()
    
    aperture = CircularAperture(positionslist, r=4) #2*FWHM
    annulus_aperture = CircularAnnulus(positionslist, r_in=6, r_out=8)#4-5*FWHM+2*FWHM
    apers = [aperture, annulus_aperture]
    
    phot_table = aperture_photometry(img, apers)
    bkg_mean = phot_table['aperture_sum_1'] / annulus_aperture.area
    bkg_sum = bkg_mean * aperture.area
    final_sum = phot_table['aperture_sum_0'] - bkg_sum 
    SNvalue = final_sum/np.sqrt(final_sum+bkg_sum)
    magstar = 25 - 2.5*np.log10(abs(final_sum/1))
    
    #posSNvalue = np.column_stack((positions, SNvalue))  
    magstarSNvalue = np.column_stack((magstar, SNvalue))
    posmagstarSNvalue = np.column_stack((positions, magstarSNvalue))
    
    return posmagstarSNvalue,magstarSNvalue

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
    
    

posSNvalue1, magstarSNvalue = SignalNoise(position, imgdata1)
plt.figure(1)
plt.plot(magstarSNvalue[:,0], 1/magstarSNvalue[:,1],'.')

hang,lie = posSNvalue1.shape
posmagsntemp = []
for i in range(hang):
    if 1/posSNvalue1[i,3]>0 and posSNvalue1[i,2]<18:
        posmagsntemp.append(posSNvalue1[i])
       
arraypms = np.array(posmagsntemp)    
ERROR = 1/arraypms[:,3]   
plt.figure(2)
plt.plot(arraypms[:,2], ERROR ,'.')  
plt.xlabel('mag',fontsize=14)
plt.ylabel('error',fontsize=14)  

np.savetxt('pms.txt', arraypms)  
positions1 = arraypms[:,0:2]  
apertures1 = CircularAperture(positions1, r=5.)
displayimage(imgdata1, 1, 3)
apertures1.plot(color='blue', lw=1.5, alpha=0.5)