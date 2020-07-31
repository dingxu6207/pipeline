# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 17:41:11 2020

@author: dingxu
"""

import numpy as np
from astropy.table import Table
from photutils.datasets import make_noise_image
from photutils.datasets import make_gaussian_sources_image
import matplotlib.pyplot as plt

from photutils.detection import IRAFStarFinder
from photutils.psf import IntegratedGaussianPRF, DAOGroup
from photutils.background import MMMBackground, MADStdBackgroundRMS
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm

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


sigma_psf = 2.0
sources = Table()
sources['flux'] = [700, 800, 700, 800]
sources['x_mean'] = [12, 57, 62, 87]
sources['y_mean'] = [15, 55, 60, 90]
sources['x_stddev'] = sigma_psf*np.ones(4)
sources['y_stddev'] = sources['x_stddev']
sources['theta'] = [0, 0, 0, 0]
sources['id'] = [1, 2, 3, 4]
tshape = (132, 132)

image = (make_gaussian_sources_image(tshape, sources) +
          make_noise_image(tshape, distribution='poisson', mean=6.,
                           random_state=1) +
         make_noise_image(tshape, distribution='gaussian', mean=0.,
                           stddev=2., random_state=1))
          


bkgrms = MADStdBackgroundRMS()
std = bkgrms(image)
iraffind = IRAFStarFinder(threshold=3.5*std,
                          fwhm=sigma_psf*gaussian_sigma_to_fwhm,
                           minsep_fwhm=0.01, roundhi=5.0, roundlo=-5.0,
                           sharplo=0.0, sharphi=2.0)
daogroup = DAOGroup(2.0*sigma_psf*gaussian_sigma_to_fwhm)
mmm_bkg = MMMBackground()
fitter = LevMarLSQFitter()
psf_model = IntegratedGaussianPRF(sigma=sigma_psf)



from photutils.psf import IterativelySubtractedPSFPhotometry
photometry = IterativelySubtractedPSFPhotometry(finder=iraffind,
                                                 group_maker=daogroup,
                                                 bkg_estimator=mmm_bkg,
                                                 psf_model=psf_model,
                                                 fitter=LevMarLSQFitter(),
                                                 niters=1, fitshape=(11,11))
result_tab = photometry(image=image)
residual_image = photometry.get_residual_image()  
        
displayimage(image, 3, 0)
plt.plot(12, 15, '*')
displayimage(residual_image, 3, 1)


from photutils.psf import BasicPSFPhotometry
psf_model.x_0.fixed = True
psf_model.y_0.fixed = True
#sources['x_mean'] = [12]
#sources['y_mean'] = [15]
pos = Table(names=['x_0', 'y_0'], data=[sources['x_mean'],sources['y_mean']])
photometry = BasicPSFPhotometry(group_maker=daogroup,
                                bkg_estimator=mmm_bkg,
                                 psf_model=psf_model,
                                fitter=LevMarLSQFitter(),
                                 fitshape=(11,11))

result_tab = photometry(image=image, init_guesses=pos)
residual_image = photometry.get_residual_image()
displayimage(residual_image, 3, 2)
print(result_tab['flux_fit'])
