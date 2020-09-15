# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:15:05 2020

@author: dingxu
"""

import numpy as np

location0 = np.loadtxt('location0.txt')

location1 = np.loadtxt('location1.txt')


def sourcephotometry(targetx, targety, sumpho, threshold=5):
    hang,lie = sumpho.shape    
    for i in range(hang):
        delt = np.sqrt((targetx - sumpho[i][0])**2+(targety - sumpho[i][1])**2)
        if delt < threshold:
            print(sumpho[i])
            

zuihou =  sourcephotometry(270,428,location1)   
kaishi =  sourcephotometry(337,394,location0)         