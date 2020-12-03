# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 22:50:30 2020

@author: dingxu
"""
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import cv2
import math

src_pts0 = np.loadtxt('src0.txt')
dst_pts0 = np.loadtxt('dst0.txt')

src_pts1 = np.loadtxt('src1.txt')
dst_pts1 = np.loadtxt('dst1.txt')

src_pts2 = np.loadtxt('src2.txt')
dst_pts2 = np.loadtxt('dst2.txt')

src_pts3 = np.loadtxt('src3.txt')
dst_pts3 = np.loadtxt('dst3.txt')

#src_pts4 = np.loadtxt('src4.txt')
#dst_pts4 = np.loadtxt('dst4.txt')


src_pts = np.row_stack((src_pts0, src_pts1))
src_pts = np.row_stack((src_pts, src_pts2))
src_pts = np.row_stack((src_pts, src_pts3))
#src_pts = np.row_stack((src_pts, src_pts4))

dst_pts = np.row_stack((dst_pts0, dst_pts1))
dst_pts = np.row_stack((dst_pts, dst_pts2))
dst_pts = np.row_stack((dst_pts, dst_pts3))
#dst_pts = np.row_stack((dst_pts, dst_pts4))

#半高全宽和匹配数目修改即可
fitsname1 = 'E:\\shunbianyuan\\dataxingtuan\\berkeley99\\'+'d4738787L018m000.fit'
fitsname2 = 'E:\\shunbianyuan\\dataxingtuan\\berkeley99\\'+'d4738787L018m001.fit'
onehdu = fits.open(fitsname1)
imgdata1 = onehdu[0].data  #hdu[0].header

copydata1 = np.copy(imgdata1)
imgdata1 = np.float32(copydata1)
#imgdata1 = np.rot90(imgdata1)
#imgdata1 = np.rot90(imgdata1)
oneimgdata = imgdata1
hang1,lie1 = oneimgdata.shape

twohdu = fits.open(fitsname2)
imgdata2 = twohdu[0].data  #hdu[0].header
#imgdata2 = np.rot90(imgdata2)
copydata2 = np.copy(imgdata2)
imgdata2 = np.float32(copydata2)
twoimgdata = imgdata2
hang2,lie2 = twoimgdata.shape


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
    #plt.savefig(str(i)+'.jpg')
    
displayimage(oneimgdata,3,0)
displayimage(twoimgdata,3,1)

hmerge = np.hstack((oneimgdata, twoimgdata)) #水平拼接
displayimage(hmerge, 1, 2)


lengthdata,liedata = src_pts.shape

for i in range(0, lengthdata):
    x10 = src_pts[i][0]
    y10 = src_pts[i][1]
    
    x11 = dst_pts[i][0]
    y11 = dst_pts[i][1]
    
    plt.plot([x10,x11+lie1],[y10,y11],linewidth = 0.8)  


H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)    
newimg = cv2.warpPerspective(oneimgdata, H, (lie1,hang1))

displayimage(newimg, 0.1, 3) 
minusimg = np.float32(newimg) - np.float32(imgdata2)  
displayimage(minusimg, 3, 4) 
displayimage(newimg, 1, 5)  
print(H)


tempmatrix = np.zeros((3,1),dtype = np.float64)
tempmatrix[2] = 1
deltemp = []
newsrc = []
for j in range(lengthdata):
    tempmatrix[0] = src_pts[j][0]
    tempmatrix[1] = src_pts[j][1]
    
    result = np.dot(H,tempmatrix)
    
    rx11 = result[0]/result[2]
    ry11 = result[1]/result[2]
        
    delcha = math.sqrt((rx11-dst_pts[j][0])**2 + (ry11-dst_pts[j][1])**2)
    deltemp.append(delcha)


plt.figure(6)
setlist = list(set(deltemp))
arraylist = np.array(setlist)
arraylist = arraylist[arraylist<2000.0]
plt.plot(arraylist, '.')
print(np.mean(arraylist))
plt.xlabel('count',fontsize=14)
plt.ylabel('delpixel',fontsize=14)