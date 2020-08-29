# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 11:13:39 2020

@author: dingxu
"""
import pandas as pd
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from photutils import CircularAperture
from itertools import combinations
import math
import itertools
from time import time
import cv2

file = 'E:\\shunbianyuan\\phometry\\data\\'+'dx.tsv'
df = pd.read_csv(file,sep=';')

#print(df.head())

npgaia = df.as_matrix()
#npgaia = npgaia[:,0:3]

listgaia = npgaia.tolist()

#listgaia.sort(key=lambda x:x[2],reverse=False)



#20190603132720Auto.fit
file  = 'ftboYFBc110170.fits'
path = 'E:\\shunbianyuan\\phometry\\todingx\\origindata\\'
filename = path+file
fitshdu = fits.open(filename)
data = fitshdu[0].data
fitsdata = np.copy(data)

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

def findsource(img):    
    mean, median, std = sigma_clipped_stats(img, sigma=3.0) 
    daofind = DAOStarFinder(fwhm = 4, threshold=5.*std)
    sources = daofind(img - median)

    #tezhen = np.transpose((sources['sharpness'], sources['roundness1'],sources['flux']))
    #print(sources[0])
    tezhen = np.transpose((sources['xcentroid'], sources['ycentroid']))
    posiandmag = np.transpose((sources['xcentroid'], sources['ycentroid'],sources['flux']))

    return tezhen,posiandmag.tolist()    
    

positions1,posiandmag1 =  findsource(fitsdata)
posiandmag1.sort(key=lambda x:x[2],reverse=True)


start = time()
print("Start: " + str(start))

apertures1 = CircularAperture(positions1, r=6.)   
displayimage(fitsdata, 1 , 0)
apertures1.plot(color='blue', lw=1.5, alpha=0.5)




lenstar1 = 30
lenstar2 = 60
lenstar = 0
posiandmag1 = posiandmag1[lenstar:lenstar1+lenstar]
posiandmag2 = listgaia[lenstar:lenstar2+lenstar]

sanjiao1 = list(combinations(posiandmag1,3))
sanjiao2 = list(combinations(posiandmag2,3))

def julisanjiao(sanjiao1,i):
    x1 = sanjiao1[i][0][0]
    y1 = sanjiao1[i][0][1]
    
    x2 = sanjiao1[i][1][0]
    y2 = sanjiao1[i][1][1]
    
    x3 = sanjiao1[i][2][0]
    y3 = sanjiao1[i][2][1]
    
    datadis1 = ((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
    dS1S2 = math.sqrt(datadis1)
    
    datadis2 = ((x1-x3)*(x1-x3)+(y1-y3)*(y1-y3))
    dS1S3 = math.sqrt(datadis2)
    
    datadis3 = ((x2-x3)*(x2-x3)+(y2-y3)*(y2-y3))
    dS2S3 = math.sqrt(datadis3)
    
    dianchen1 = (x2-x1)*(x3-x1)+(y2-y1)*(y3-y1)
    theta1 = dianchen1/(dS1S2*dS1S3)
    
    dianchen2 = (x1-x2)*(x3-x2)+(y1-y2)*(y3-y2)
    theta2 = dianchen2/(dS1S2*dS2S3)
    
    dianchen3 = (x1-x3)*(x2-x3)+(y1-y3)*(y2-y3)
    theta3 = dianchen3/(dS1S3*dS2S3)
       
    return [[x1,y1],[x2,y2],[x3,y3],[dS1S2,dS1S3,dS2S3,dianchen1]]

lensan1 = len(sanjiao1)
temp1 = [julisanjiao(sanjiao1,i) for i in range (0,lensan1)]

    
lensan2 = len(sanjiao2)
temp2 = [julisanjiao(sanjiao2,i) for i in range (0,lensan2)]    


pitemp1 = []
pitemp2 = []   
count = 0 

for i in itertools.product(temp1, temp2):
    oneju0 = i[0][3][0]
    oneju1 = i[0][3][1]
    oneju2 = i[0][3][2]
    dotmul1 = i[0][3][3]
    oneab = oneju0/oneju1
    onebc = oneju1/oneju2
    oneca = oneju2/oneju0
    oneac = oneju0/oneju2
        
    twoju0 = i[1][3][0]
    twoju1 = i[1][3][1]
    twoju2 = i[1][3][2]
    dotmul2 = i[1][3][3]
    twoab = twoju0/twoju1
    twobc = twoju1/twoju2
    twoca = twoju2/twoju0
    twoac = twoju0/twoju2
        
    pan1 = abs(oneab-twoab)
    pan2 = abs(onebc-twobc)
    pan3 = abs(oneca-twoca)
    pan4 = abs(oneac-twoac)
    pan5 = abs(dotmul1-dotmul2)
        
    
    if (pan1 < 0.0001)and(pan2<0.0001)and(pan3<0.0001):
        pitemp1.append(i[0])
        pitemp2.append(i[1])
        count = count+1    

        
stop = time()
print("Stop: " + str(stop))
print(str(stop-start) + "秒") 

srckp1 = []
srckp2 = []
for i in range(0,count):
    for j in range(0,3):
        
        x10 = pitemp1[i][j][0]
        x11 = pitemp2[i][j][0]
            
        y10 = pitemp1[i][j][1]
        y11 = pitemp2[i][j][1]
        
        srckp1.append(x10)
        srckp1.append(y10)
        srckp2.append(x11)
        srckp2.append(y11)
        src_pts = np.float32(srckp1).reshape(-1,2)
        dst_pts = np.float32(srckp2).reshape(-1,2)
        
        
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0) 
pipeidata = np.hstack((src_pts,dst_pts))
np.savetxt('pipeidata.txt', pipeidata)
print(H)       
