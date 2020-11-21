# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 21:46:50 2020

@author: dingxu
"""

import numpy as np
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import mixture
from matplotlib.pyplot import MultipleLocator
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

np.random.seed(8)

data = np.loadtxt('Gaiadata.txt')
#data = data[data[:,2]>0]
#data = data[data[:,2]<1]

data = data[data[:,3]<15]
data = data[data[:,3]>-15]

data = data[data[:,4]<15]
data = data[data[:,4]>-15]

X = np.copy(data[:,0:5])


k = 2    #聚类的级别
iteration = 1000    #剧烈最大循环次数

#data_zs = 1.0 *(X-X.mean())/X.std()    #数据标准化
X = StandardScaler().fit_transform(X)
data_zs = np.copy(X)
 
model = KMeans(n_clusters=k, n_jobs=6, max_iter=iteration)    #分为k类，并发数4
model.fit(data_zs)    #开始聚类，训练模型
datalables = model.predict(data_zs) 
centers = model.cluster_centers_

distance1 = np.sqrt(((data_zs-centers[0]))**2)
onesdata = np.ones((5,1))
sumdistance1 = np.dot(distance1,onesdata)

distance2 = np.sqrt(((data_zs-centers[1]))**2)
onesdata = np.ones((5,1))
sumdistance2 = np.dot(distance2, onesdata)


datacol = np.column_stack((data ,sumdistance1))
datacol = np.column_stack((datacol ,sumdistance2))
datapro = np.column_stack((datacol ,datalables))

lowdata = datapro[datapro[:,9] < datapro[:,8]]
#lowdata = lowdata[lowdata[:,8]<5]

highdata = datapro[datapro[:,9] > datapro[:,8]]
highdata = highdata[highdata[:,9]<3.8]


plt.figure(1)
plt.scatter(lowdata[:,3], lowdata[:,4], marker='o', color='grey',s=5.0)
plt.scatter(highdata[:,3], highdata[:,4], marker='o', color='lightcoral',s=5.0)
plt.xlabel('pmRA',fontsize=14)
plt.ylabel('pmDEC',fontsize=14)
plt.xlim((-15,15))
plt.ylim((-15,15))

plt.figure(2)
plt.plot(datapro[:,9],'.')
plt.plot(datapro[:,8],'.')


plt.figure(3)
highdataGmag = highdata[:,5]
highdataBPRP = highdata[:,6]-highdata[:,7]

plt.xlim((-1,4))
plt.ylim((10,22))

plt.scatter(highdataBPRP, highdataGmag, marker='o', color='lightcoral',s=5)
x_major_locator = MultipleLocator(1)
plt.xlabel('BP-RP',fontsize=14)
#plt.xlabel('G-RP',fontsize=14)
plt.ylabel('Gmag',fontsize=14)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反