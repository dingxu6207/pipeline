# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 15:57:14 2020

1、sessaligen
2、sessfindsatar
3、seseephometry
4、sessdisplay
@author: dingxu
"""

import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from photutils import CircularAperture, CircularAnnulus
from photutils import aperture_photometry
from astropy.time import Time
from astropy.stats import sigma_clipped_stats

filetemp = []
count = 0
oripath = 'E:\\shunbianyuan\\dataxingtuan\\alberkeley99\\'  #路径参数
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

def photometryimg(positions, img, i):
    
    positionslist = positions.tolist()
    
    aperture = CircularAperture(positionslist, r=7) #2*FWHM
    annulus_aperture = CircularAnnulus(positionslist, r_in=10, r_out=12)#4-5*FWHM+2*FWHM
    apers = [aperture, annulus_aperture]
    
    displayimage(img, 1, i) ###画图1
    aperture.plot(color='blue', lw=0.5)
    annulus_aperture.plot(color='red', lw=0.2)
    plt.pause(0.001)
    plt.clf()
    
    phot_table = aperture_photometry(img, apers)
    bkg_mean = phot_table['aperture_sum_1'] / annulus_aperture.area
    bkg_sum = bkg_mean * aperture.area
    final_sum = phot_table['aperture_sum_0'] - bkg_sum
    phot_table['residual_aperture_sum'] = final_sum       
    posflux = np.column_stack((positions, phot_table['residual_aperture_sum']))  
    #return posflux
    magstar = 25 - 2.5*np.log10(abs(final_sum/1))
    return posflux,magstar

def CircleMaskPhometry(data,location,index = 2):
    Mimgdata = np.copy(data)
    Mlocatin = np.copy(location)
    
    Mapeture = CircularAperture(Mlocatin, r=6)
    Mannuals = CircularAnnulus(Mlocatin, r_in=8., r_out=11.)
        
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
    Mapeture.plot(color='blue', lw=0.5)
    Mannuals.plot(color='red', lw=0.2)
    plt.pause(0.001)
    plt.clf()
    #Mannulus_data = Eannuals_masks[0].multiply(Mimgdata)
    #displayimage(Mannulus_data,3,index+1)
    
    flux_sum =  phot['aper_sum_bkgsub']
    magstar = 25 - 2.5*np.log10(abs(flux_sum/1))

    return Mpositionflux,magstar


def sourcephotometry(targetx, targety, sumpho, threshold=10):
    hang,lie = sumpho.shape    
    for i in range(hang):
        delt = np.sqrt((targetx - sumpho[i][0])**2+(targety - sumpho[i][1])**2)
        if delt < threshold:
            #print(sumpho[i])
            mag = 25 - 2.5*np.log10(sumpho[i][2]/1) #90曝光时间
            #print(mag)
            return sumpho[i],mag

def pltquxian(datayuan):
    data = np.array(datayuan)  
    data1 = np.copy(data)
    u = np.mean(data1)   
    std = np.std(data1)
    error = data1[np.abs(data1 - u) > 3*std]
    data_c = data1[np.abs(data1 - u) <= 3*std] 
    print( len(error))
    return data_c
 
files = 'locationmi30.txt' 
pathfile = 'E:\\shunbianyuan\\phometry\\pipelinecode\\pipeline\\LiXZ\location\\'+files
lacation = np.loadtxt(pathfile)     

fitshdu = fits.open(oripath+filetemp[1])
fitsdata = fitshdu[0].data
displayimage(fitsdata, 1, 0)

jiaoyantemp = []
startemp = []
targettemp = []
datatemp = []

#796*i:796+796*i,778*j:778+778*j
m = 3#行扫描 i = 39
n = 0#列扫描 j = 39
print(m,n)
for i in range(0, count):
    try:
        fitshdu = fits.open(oripath+filetemp[i])
        datatime = fitshdu[0].header['DATE']
        t = Time(datatime, format='isot', scale='utc')
        data = fitshdu[0].data    
        fitsdata = data[796*m:796+796*m,778*n:778+778*n]
        #posflux,magstar = photometryimg(lacation, fitsdata, 1)  
        posflux,magstar = CircleMaskPhometry(fitsdata, lacation, 1)         
        startemp.append(magstar) 
        #arraytemp = np.array(startemp).T        
        
        posflux1,mag1 = sourcephotometry(119, 94, posflux)  #比较星位置1        
        posflux2,mag2 = sourcephotometry(326, 550, posflux)  #比较星位置2
        
        #posflux3,mag3 = sourcephotometry(249, 234, posflux)   
       
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
