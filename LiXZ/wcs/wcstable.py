# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 19:49:40 2021

@author: dingxu
"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from photutils import CircularAperture
from astroquery.astrometry_net import AstrometryNet
from astropy.wcs import WCS


ast = AstrometryNet()
ast.api_key = 'vslojcwowmxjczlq'

filename = 'new-image.fits'
#path = "E:/shunbianyuan/phometry/todingx/origindata/"
#file = "ftboYFAk300222.fits"
#path = 'E:\\shunbianyuan\\dataxingtuan\\ngc7142\\'
#file = 'd4738777L016m000.fit'
#filename = path+file
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
displayimage(fitsdata,3,0)
apertures1.plot(color='blue', lw=1.5, alpha=0.5)
#FWHMplot(505,508,10,fitsdata,1)


image_width,image_height = fitsdata.shape
#image_height = 3945
sources1.sort('flux')
wcs_header = ast.solve_from_source_list(sources1['xcentroid'], sources1['ycentroid'],
                                        image_width, image_height,
                                        solve_timeout=200)

print(wcs_header)


wcs_gamcas = WCS(wcs_header)
print(wcs_gamcas)

#pixcrd = np.array([[0, 0], [24, 38], [45, 98]], np.float_)
pixcrd = np.copy(positions1)
world = wcs_gamcas.wcs_pix2world(pixcrd, 1)
print(world)