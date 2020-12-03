# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:38:30 2020

@author: dingxu
"""
import numpy as np
import math
import matplotlib.pyplot as plt

src_pts = np.loadtxt('src.txt')
dst_pts = np.loadtxt('dst.txt')

src_pts1 = np.loadtxt('src1.txt')
dst_pts1 = np.loadtxt('dst1.txt')

#src_pts = src_pts1[:,0:2]
#dst_pts = dst_pts1[:,0:2]

lenpipei,lie = src_pts.shape

H = [[ 1.00008578e+00, -1.20967723e-03,  2.94079973e+00],
    [ 1.24003836e-03,  1.00009134e+00, -2.74102169e+00],
    [ 2.84827755e-08,  3.99474805e-09,  1.00000000e+00]]


tempmatrix = np.zeros((3,1),dtype = np.float64)
tempmatrix[2] = 1
deltemp = []
newsrc = []
for j in range(lenpipei):
    tempmatrix[0] = src_pts[j][0]
    tempmatrix[1] = src_pts[j][1]
    
    result = np.dot(H,tempmatrix)
    
    rx11 = result[0]/result[2]
    ry11 = result[1]/result[2]
        
    delcha = math.sqrt((rx11-dst_pts[j][0])**2 + (ry11-dst_pts[j][1])**2)
    deltemp.append(delcha)


plt.figure(5)
setlist = list(set(deltemp))
arraylist = np.array(setlist)
arraylist = arraylist[arraylist<100.0]
plt.plot(arraylist, '.')
print(np.mean(arraylist))
plt.xlabel('count',fontsize=14)
plt.ylabel('delpixel',fontsize=14)