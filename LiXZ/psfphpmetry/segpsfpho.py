# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 15:55:35 2020

@author: dingxu
"""

import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from photutils import CircularAperture, CircularAnnulus
from photutils import aperture_photometry
from astropy.time import Time

from astropy.table import Table
from photutils.psf import IntegratedGaussianPRF, DAOGroup
from photutils.background import MMMBackground
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm
from photutils.psf import BasicPSFPhotometry

filetemp = []
count = 0
oripath = 'E:\\shunbianyuan\\dataxingtuan\\alberkeley99\\' #路径参数
for root, dirs, files in os.walk(oripath):
   for file in files:
       if (file[-4:] == '.fit'):
           count = count+1
           filetemp.append(file)
       
       
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
    #plt.plot(376.82, 137.232, '*')
    #plt.savefig(oripath+str(i)+'.jpg')


def sourcephotometry(targetx, targety, sumpho, threshold=3):
    hang,lie = sumpho.shape    
    for i in range(hang):
        delt = np.sqrt((targetx - sumpho[i][0])**2+(targety - sumpho[i][1])**2)
        if delt < threshold:
            #print(sumpho[i])
            mag = 25 - 2.5*np.log10(sumpho[i][2]/1) #90曝光时间
            #print(mag)
            return sumpho[i],mag
        
def photomyPSF(imgdata, position,sigma):
    PSFdata = np.copy(imgdata)
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
                                    fitshape=(11,11))

    result_tab = photometry(image=PSFdata, init_guesses=pos) 
    positionflux = np.transpose((result_tab['x_fit'], result_tab['y_fit'],  result_tab['flux_fit']))
    
    magstar = 25 - 2.5*np.log10(abs(result_tab['flux_fit']/1))
    return positionflux,magstar    

files = 'locationmi41.txt' 
pathfile = 'E:\\shunbianyuan\\phometry\\pipelinecode\\pipeline\\LiXZ\location\\'+files
lacation = np.loadtxt(pathfile)     

fitshdu = fits.open(oripath+filetemp[0])
fitsdata = fitshdu[0].data
displayimage(fitsdata, 1, 0)


jiaoyantemp = []
startemp = []
targettemp = []
datatemp = []

m = 4#行扫描 i = 39
n = 1#列扫描 j = 39
#796*i:796+796*i,778*j:778+778*j

for i in range(0, count):
    try:
        fitshdu = fits.open(oripath+filetemp[i])
        datatime = fitshdu[0].header['DATE']
        t = Time(datatime, format='isot', scale='utc')
        data = fitshdu[0].data    
        fitsdata = data[796*m:796+796*m,778*n:778+778*n]  
        
        posflux,magstar = photomyPSF(fitsdata, lacation, 0.90)        
        startemp.append(magstar) 
        #arraytemp = np.array(startemp).T 
       
        
        posflux1,mag1 = sourcephotometry(532, 148, posflux)  #比较星位置1 
         
        posflux2,mag2 = sourcephotometry(237, 68, posflux)  #比较星位置2
        
        #posflux3,mag3 = sourcephotometry(285, 363, posflux)   
       
        jiaoyan = mag1-mag2 
        #target = mag3 - mag1
                
        jiaoyantemp.append(jiaoyan)
        #targettemp.append(target)
        datatemp.append(t.jd)
              
        
        
        print('ok')
    except:
        print('error!!!')
    
arraytemp = np.array(startemp).T
starlight = np.hstack((posflux[:,0:2], arraytemp)) 
np.savetxt('starlight.txt', starlight)   
#jiaoyandata = pltquxian(jiaoyan)       
plt.figure(2)
plt.plot(datatemp,jiaoyantemp,'.')
arraytime = np.array(datatemp)
np.savetxt('datatime.txt', arraytime)
