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
    
    aperture = CircularAperture(positionslist, r=4.5) #2*FWHM
    annulus_aperture = CircularAnnulus(positionslist, r_in=5.5, r_out=6.5)#4-5*FWHM+2*FWHM
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

def sourcephotometry(targetx, targety, sumpho, threshold=3):
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
 
lacation = np.loadtxt('location.txt')     

fitshdu = fits.open(oripath+filetemp[0])
fitsdata = fitshdu[0].data
displayimage(fitsdata, 1, 0)

jiaoyantemp = []
startemp = []
targettemp = []
datatemp = []

m = 0#行扫描 i = 39
n = 5#列扫描 j = 39
for i in range(0, count):
    try:
        fitshdu = fits.open(oripath+filetemp[i])
        datatime = fitshdu[0].header['DATE']
        t = Time(datatime, format='isot', scale='utc')
        data = fitshdu[0].data    
        fitsdata = data[398*m:398+398*m,389*n:389+389*n]
        posflux,magstar = photometryimg(lacation, fitsdata, 1)           
        startemp.append(magstar) 
        arraytemp = np.array(startemp).T        
        
        posflux1,mag1 = sourcephotometry(282, 288, posflux)  #比较星位置1        
        posflux2,mag2 = sourcephotometry(187, 267, posflux)  #比较星位置2
        
        posflux3,mag3 = sourcephotometry(186, 357, posflux)   
       
        jiaoyan = mag1-mag2 
        target = mag3 - mag1
                
        jiaoyantemp.append(jiaoyan)
        targettemp.append(target)
        datatemp.append(t.jd)
                
        
        
        print('ok')
    except:
        print('error!!!')
    
#datatemp[:] = [x - 2458776 for x in datatemp]
plt.figure(2)
plt.plot(datatemp, jiaoyantemp,'.')
plt.xlabel('JD',fontsize=14)
plt.ylabel('mag',fontsize=14)

plt.figure(3)
plt.plot(datatemp,targettemp,'.')
plt.xlabel('JD',fontsize=14)
plt.ylabel('mag',fontsize=14)
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向

templist = []
templist.append(datatemp)
templist.append(targettemp)
tempmatrix = np.array(templist)
np.savetxt('datamag.txt', tempmatrix)

