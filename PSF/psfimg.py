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

fitsname1 = 'E:\\shunbianyuan\\phometry\\todingx\\origindata\\'+'ftboYFAk010148.fits'

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

xp = position[10][0]
yp = position[10][1]
displayimage(imgdata1,1,0)
plt.plot(xp,yp,'*')

sigma_psf = 2.23
daogroup = DAOGroup(2.0*sigma_psf*gaussian_sigma_to_fwhm)
mmm_bkg = MMMBackground()
fitter = LevMarLSQFitter()
psf_model = IntegratedGaussianPRF(sigma=sigma_psf)


from photutils.psf import BasicPSFPhotometry
sources = Table()
sources['x_mean'] = [xp]
sources['y_mean'] = [yp]
#sources['x_mean'] = position[:,0]
#sources['y_mean'] = position[:,1]
psf_model.x_0.fixed = True
psf_model.y_0.fixed = True
pos = Table(names=['x_0', 'y_0'], data=[sources['x_mean'],sources['y_mean']])
photometry = BasicPSFPhotometry(group_maker=daogroup,
                                bkg_estimator=mmm_bkg,
                                 psf_model=psf_model,
                                fitter=LevMarLSQFitter(),
                                 fitshape=(13,13))

result_tab = photometry(image=imgdata1, init_guesses=pos)
residual_image = photometry.get_residual_image()


print(xp,yp,result_tab['flux_fit'][0]) #flux_0  flux_fit


posflux,magstar = photometryimg(position, imgdata1, 1)
print(posflux[10])