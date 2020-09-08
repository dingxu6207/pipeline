# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 15:22:46 2020

@author: dingxu
"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from photutils import CircularAperture
from photutils import EllipticalAperture
from photutils import aperture_photometry
import math
from photutils import EllipticalAnnulus


filename = 'E:\\shunbianyuan\\newdata\\201911162230420716.fit'
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
    daofind = DAOStarFinder(fwhm=8.0, threshold=5.*std)
    sources = daofind(img - median)

    for col in sources.colnames:
        sources[col].info.format = '%.8g'  # for consistent table output
        #print(sources)

    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    positionflux = np.transpose((sources['xcentroid'], sources['ycentroid'],  sources['flux']))
    mylist = positionflux.tolist()
    
    return sources,positions,mylist


    
def displayimage(img, coff, i):
    minimg,maximg = adjustimage(img, coff)
    plt.figure(i)
    plt.imshow(img, cmap='gray', vmin = minimg, vmax = maximg)
    plt.savefig(str(i)+'.jpg')
    

def EllipticalAperturePhometry(data,location, jiao = 1.2,index = 1):
    Eimgdata = np.copy(data)
    Elocatin = np.copy(location)
    
    Eapeture = EllipticalAperture(Elocatin, a=12, b=8, theta = jiao)
    Eannuals = EllipticalAnnulus(Elocatin, a_in=16, a_out=20, b_out=14, theta = jiao)
    
    displayimage(Eimgdata,3,index)
    Eapeture.plot(color='blue', lw=1.5, alpha=0.5)
    Eannuals.plot(color='red', lw=1.5, alpha=0.5)
    
    apers = [Eapeture, Eannuals]
    phot_table = aperture_photometry(Eimgdata, apers)
    
    bkg_mean = phot_table['aperture_sum_1'] / Eannuals.area
    bkg_sum = bkg_mean * Eapeture.area
    final_sum = phot_table['aperture_sum_0'] - bkg_sum
    phot_table['residual_aperture_sum'] = final_sum
    
    Epositionflux = np.transpose((phot_table['xcenter'], phot_table['ycenter'],  phot_table['residual_aperture_sum']))

    return Epositionflux
    
def EllipticalMaskPhometry(data,location, jiao = 1.2,index = 2):
    Eimgdata = np.copy(data)
    Elocatin = np.copy(location)
    
    Eapeture = EllipticalAperture(Elocatin, a=12, b=8, theta = jiao)
    Eannuals = EllipticalAnnulus(Elocatin, a_in=16, a_out=20, b_out=14, theta = jiao)
        
    Eannuals_masks = Eannuals.to_mask(method='center')
    
    bkg_median = []
    for mask in Eannuals_masks:
        Eannuals_data = mask.multiply(Eimgdata)
        Eannulus_data_1d = Eannuals_data[mask.data > 0]
        meandata,median_sigclip,_ = sigma_clipped_stats(Eannulus_data_1d)
        bkg_median.append(median_sigclip) 
        
    bkg_median = np.array(bkg_median)     
    phot = aperture_photometry(Eimgdata, Eapeture)
    phot['annulus_median'] = bkg_median
    phot['aper_bkg'] = bkg_median * Eapeture.area
    phot['aper_sum_bkgsub'] = phot['aperture_sum'] - phot['aper_bkg']
    
    EMpositionflux = np.transpose((phot['xcenter'], phot['ycenter'],  phot['aper_sum_bkgsub']))
    
    displayimage(Eimgdata,3,index)
    Eapeture.plot(color='blue', lw=1.5, alpha=0.5)
    Eannuals.plot(color='red', lw=1.5, alpha=0.5)
    Eannulus_data = Eannuals_masks[0].multiply(Eimgdata)
    displayimage(Eannulus_data,3,index+1)

    return EMpositionflux
    
sources1,positions1,mylist1 =  findsource(fitsdata)
apertures1 = CircularAperture(positions1, r=10.)
displayimage(fitsdata,3,0)
apertures1.plot(color='blue', lw=1.5, alpha=0.5)


x1,y1 = 397,130
x2,y2 = 405,151

thetajiao = math.atan((y1-y2)/(x1-x2))

Eposlux = EllipticalAperturePhometry(fitsdata, positions1)
Mposflux = EllipticalMaskPhometry(fitsdata,positions1)


plt.figure(4)
plt.plot(Eposlux[:,2]-Mposflux[:,2], '*')

#plt.xlabel('Eposlux',fontsize=14)
#plt.ylabel('Mposflux',fontsize=14)




