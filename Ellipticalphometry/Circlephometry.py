# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 10:40:49 2020

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
from photutils import CircularAnnulus


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
    

def CircleAperturePhometry(data,location,index):
    Cimgdata = np.copy(data)
    Clocatin = np.copy(location)
    
    Caperture = CircularAperture(Clocatin, r=12)
    Cannulusaperture = CircularAnnulus(Clocatin, r_in=14., r_out=18.)
    
    displayimage(Cimgdata,3,index)
    Caperture.plot(color='blue', lw=1.5, alpha=0.5)
    Cannulusaperture.plot(color='red', lw=1.5, alpha=0.5)
    
    apers = [Caperture, Cannulusaperture]
    phot_table = aperture_photometry(Cimgdata, apers)
    
    bkg_mean = phot_table['aperture_sum_1'] / Cannulusaperture.area
    bkg_sum = bkg_mean * Caperture.area
    final_sum = phot_table['aperture_sum_0'] - bkg_sum
    phot_table['residual_aperture_sum'] = final_sum
    
    Cpositionflux = np.transpose((phot_table['xcenter'], phot_table['ycenter'],  phot_table['residual_aperture_sum']))

    return Cpositionflux
    
def CircleMaskPhometry(data,location,index = 2):
    Mimgdata = np.copy(data)
    Mlocatin = np.copy(location)
    
    Mapeture = CircularAperture(Mlocatin, r=12)
    Mannuals = CircularAnnulus(Mlocatin, r_in=14., r_out=18.)
        
    Eannuals_masks = Mannuals.to_mask(method='center')
    
    bkg_median = []
    for mask in Eannuals_masks:
        Eannuals_data = mask.multiply(Mimgdata)
        Eannulus_data_1d = Eannuals_data[mask.data > 0]
        meandata,median_sigclip,_ = sigma_clipped_stats(Eannulus_data_1d)
        bkg_median.append(median_sigclip) 
        
    bkg_median = np.array(bkg_median)     
    phot = aperture_photometry(Mimgdata, Mapeture)
    phot['annulus_median'] = bkg_median
    phot['aper_bkg'] = bkg_median * Mapeture.area
    phot['aper_sum_bkgsub'] = phot['aperture_sum'] - phot['aper_bkg']
    
    Mpositionflux = np.transpose((phot['xcenter'], phot['ycenter'],  phot['aper_sum_bkgsub']))
    
    displayimage(Mimgdata,3,index)
    Mapeture.plot(color='blue', lw=1.5, alpha=0.5)
    Mannuals.plot(color='red', lw=1.5, alpha=0.5)
    Mannulus_data = Eannuals_masks[0].multiply(Mimgdata)
    displayimage(Mannulus_data,3,index+1)

    return Mpositionflux
    
sources1,positions1,mylist1 =  findsource(fitsdata)
apertures1 = CircularAperture(positions1, r=12.)
displayimage(fitsdata,3,0)
apertures1.plot(color='blue', lw=1.5, alpha=0.5)

Cpositionflux = CircleAperturePhometry(fitsdata,positions1,index=1)
Mpositionflux = CircleMaskPhometry(fitsdata, positions1, 2)

plt.figure(4)
plt.plot(Cpositionflux[:,2]-Mpositionflux[:,2], '.')

