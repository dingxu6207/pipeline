# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 21:18:13 2020

@author: dingxu
"""
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

from astropy.table import Table
from photutils.psf import IntegratedGaussianPRF, DAOGroup
from photutils.background import MMMBackground
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm

from photutils import CircularAperture, CircularAnnulus
from photutils import aperture_photometry

#fitsname1 = 'E:\\shunbianyuan\\phometry\\todingx\\origindata\\'+'ftboYFAk010148.fits'
file  = '0.fits'
path = 'E:\\shunbianyuan\\dataxingtuan\\alngc7142\\'
fitsname1 = path+file

onehdu = fits.open(fitsname1)
imgdata1 = onehdu[0].data  #hdu[0].header

position = np.loadtxt('location.txt')

def photometryimg(positions, img, i):
    
    positionslist = positions.tolist()
    
    aperture = CircularAperture(positionslist, r=10) #2*FWHM
    annulus_aperture = CircularAnnulus(positionslist, r_in=20, r_out=30)#4-5*FWHM+2*FWHM
    apers = [aperture, annulus_aperture]
    
    displayimage(img, 1, i) ###画图1
    aperture.plot(color='blue', lw=0.5)
    annulus_aperture.plot(color='red', lw=0.2)
   
    phot_table = aperture_photometry(img, apers)
    print(type(phot_table))
    bkg_mean = phot_table['aperture_sum_1'] / annulus_aperture.area
    bkg_sum = bkg_mean * aperture.area
    final_sum = phot_table['aperture_sum_0'] - bkg_sum
    #phot_table['residual_aperture_sum'] = final_sum       
    posflux = np.column_stack((positions, final_sum))  
    #return posflux
    magstar = 25 - 2.5*np.log10(abs(final_sum/1))
    return posflux,magstar

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
    
 
from photutils.psf import BasicPSFPhotometry
def photomyPSF(psflocation,sigma):
    sigma_psf = sigma
    daogroup = DAOGroup(2.0*sigma_psf*gaussian_sigma_to_fwhm)
    mmm_bkg = MMMBackground()
    #fitter = LevMarLSQFitter()
    psf_model = IntegratedGaussianPRF(sigma=sigma_psf)

    sources = Table()
    #x0 = psflocation[10][0]
    #y0 = psflocation[10][1]
    
    #x1 = psflocation[9][0]
    #y1 = psflocation[9][1]
    
    #sources['x_mean'] = [x0,x1]#position[:,0].T
    #sources['y_mean'] = [y0,y1]
    sources['x_mean'] = position[:,0].T
    sources['y_mean'] = position[:,1].T

    psf_model.x_0.fixed = True
    psf_model.y_0.fixed = True
    pos = Table(names=['x_0', 'y_0'], data=[sources['x_mean'],sources['y_mean']])
    photometry = BasicPSFPhotometry(group_maker=daogroup,
                                    bkg_estimator=mmm_bkg,
                                    psf_model=psf_model,
                                    fitter=LevMarLSQFitter(),
                                    fitshape=(13,13))

    result_tab = photometry(image=imgdata1, init_guesses=pos)
    #print(result_tab)
    
    positionflux = np.transpose((result_tab['x_fit'], result_tab['y_fit'],  result_tab['flux_fit']))

   # print(result_tab['flux_fit']) #flux_0  flux_fit
    #return result_tab
    return positionflux

def photomyPSFmodel(imgdata, position,sigma):
    imgdata1 = np.copy(imgdata)
    sigma_psf = sigma
    daogroup = DAOGroup(2.0*sigma_psf*gaussian_sigma_to_fwhm)
    mmm_bkg = MMMBackground()
    #fitter = LevMarLSQFitter()
    psf_model = IntegratedGaussianPRF(sigma=sigma_psf)

    sources = Table()

    sources['x_mean'] = position[:,0].T
    sources['y_mean'] = position[:,1].T

    psf_model.x_0.fixed = True
    psf_model.y_0.fixed = True
    pos = Table(names=['x_0', 'y_0'], data=[sources['x_mean'],sources['y_mean']])
    photometry = BasicPSFPhotometry(group_maker=daogroup,
                                    bkg_estimator=mmm_bkg,
                                    psf_model=psf_model,
                                    fitter=LevMarLSQFitter(),
                                    fitshape=(13,13))

    result_tab = photometry(image=imgdata1, init_guesses=pos)
    print(result_tab)    
    positionflux = np.transpose((result_tab['x_fit'], result_tab['y_fit'],  result_tab['flux_fit']))
    
    magstar = 25 - 2.5*np.log10(abs(result_tab['flux_fit']/1))
    return positionflux,magstar



#fluxtable =  photomyPSF(position,sigma=2.23)
fluxtable =  photomyPSF(position,0.89)
positionflux,magstar = photomyPSFmodel(imgdata1, position, sigma=0.89)

#posflux,magstar = photometryimg(position, imgdata1, 1)
#print(posflux[10][2]-posflux[9][2])

#displayimage(imgdata1,1,0)
#plt.plot(posflux[7][0], posflux[7][1], '*')