# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 21:45:25 2020

@author: dingxu
"""

import numpy as np
from sklearn import decomposition 
import matplotlib.pyplot as plt
import os
from pandas.core.frame import DataFrame
from sklearn.preprocessing import StandardScaler


path = 'E:\\shunbianyuan\\data\\kepler\\KIC_name\\'

#pca = PCA(n_components=100)

#newdata =  pca.fit(data[:,1].T)

'''
plt.figure(0)
plt.plot(data[:,0], data[:,1], '.')
plt.plot(data1[:,0], data1[:,1], '.')
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
'''


filetemp = []
for root, dirs, files in os.walk(path):
   for file in files:
       strfile = os.path.join(root, file)
       #print(strfile)
       filetemp.append(strfile)
       
       

lengthfile = len(filetemp)
tempflux = []

for i in range(lengthfile):
    data = np.loadtxt(filetemp[i])
    data = data[:,1].T
    tempflux.append(data)
    
df = DataFrame(tempflux)

df = df.fillna(0)


df = StandardScaler().fit_transform(df)
arraydata = np.array(df)

pca = decomposition.PCA(n_components=0.9)
copydata = np.copy(arraydata)
pca.fit(copydata)
pcadata = pca.transform(copydata)

plt.figure(0)
plt.plot(arraydata[0,:], '.')
plt.plot(arraydata[15,:], '.')
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向