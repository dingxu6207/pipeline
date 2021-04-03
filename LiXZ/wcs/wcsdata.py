# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 01:10:36 2021

@author: dingxu
"""

from astropy.wcs import WCS
from astropy.io import fits
import numpy as np
from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from photutils import CircularAperture
import matplotlib.pyplot as plt

#path = 'E:\\shunbianyuan\\phometry\\todingx\\origindata\\'
#file = 'ftboYFAk300222.fits'
#filename = path+file


#w = WCS(filename)

w = WCS('new-image.fits')

'''
lon, lat = w.all_pix2world(1968.054, 1990.535, 0)

print(lon, lat)


pixcrd = np.array([[0, 0], [24, 38], [45, 98]], np.float_)
world = w.wcs_pix2world(pixcrd, 1)
print(world)
'''
filename = 'new-image.fits'
fitshdu = fits.open(filename)
fitsdata = fitshdu[0].data

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



def findsource(img):    
    mean, median, std = sigma_clipped_stats(img, sigma=3.0)
    daofind = DAOStarFinder(fwhm=2, threshold=5.*std)
    sources = daofind(img - median)

    for col in sources.colnames:
        sources[col].info.format = '%.8g'  # for consistent table output
        #print(sources)

    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    positionflux = np.transpose((sources['xcentroid'], sources['ycentroid'],  sources['flux']))
    mylist = positionflux.tolist()
    
    return sources,positions,mylist

def FWHMplot(x0,y0,width,img,i):
    x0 = int(x0)
    y0 = int(y0)
    pixlist = []
    for i in range((y0-width),(y0+width)):
        pixlist.append(img[x0,i])
    
    plt.figure(i)
    plt.plot(pixlist)
    plt.grid(color='r',linestyle='--',linewidth=2)
    
def displayimage(img, coff, i):
    minimg,maximg = adjustimage(img, coff)
    plt.figure(i)
    plt.imshow(img, cmap='gray', vmin = minimg, vmax = maximg)
    plt.savefig(str(i)+'.jpg')
    
sources1,positions1,mylist1 =  findsource(fitsdata)
apertures1 = CircularAperture(positions1, r=10.)
displayimage(fitsdata,1,0)
plt.plot(positions1[5921][0], positions1[5921][1], '*')
#apertures1.plot(color='blue', lw=1.5, alpha=0.5)

world = w.wcs_pix2world(positions1,1)
#print(world)