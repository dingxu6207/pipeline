# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 17:23:48 2020

@author: dingxu
"""


#RA_ICRS;DE_ICRS;Plx;pmRA;pmDE;Gmag;BPmag;RPmag


import numpy as np
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import mixture



#np.random.seed(7)

data = np.loadtxt('Gaiadata.txt')
data = data[data[:,2]>0]
data = data[data[:,2]<1]

data = data[data[:,3]<15]
data = data[data[:,3]>-15]

data = data[data[:,4]<15]
data = data[data[:,4]>-15]

#data = data[data[:,7]<18]


X = np.copy(data[:,0:5])

#zsx = 1.0*(X-X.mean())/(X.std())
'''
hang,lie = X.shape
for i in range(0,hang):
    X[i:,] = (X[i:,]-np.min(X[i:,]))/(np.max(X[i:,]) - np.min(X[i:,]))
    
''' 
   
#zsx = X
#model = KMeans(n_clusters = 3)
#model = AgglomerativeClustering(n_clusters = 2)
#model.fit(zsx)
#print(model.labels_)
#m = model.labels_

#predict_labels = model.predict(zsx)


ZSX = np.copy(X)
clst = mixture.GaussianMixture(n_components = 2)

predicted_lables = clst.fit_predict(ZSX)

prodata = clst.predict_proba(ZSX)

r1 = pd.Series(predicted_lables).value_counts()

print(r1)


Gmag = data[:,5]
BPRP = data[:,6]-data[:,7] 

plt.figure(0)
plt.plot(BPRP, Gmag, '.')
plt.xlabel('BP-RP',fontsize=14)
plt.ylabel('Gmag',fontsize=14)
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向


datalable = np.column_stack((data ,predicted_lables))

cydata = datalable[datalable[:,8]==1]
waidata = datalable[datalable[:,8]==0]


datapro = np.column_stack((data ,prodata))
highdata = datapro[datapro[:,9] > datapro[:,8]]
highdata = highdata[highdata[:,9]>0.985]


plt.figure(1)
plt.scatter(waidata[:,3], waidata[:,4], marker='o', color='grey',s=10.0)
plt.scatter(highdata[:,3], highdata[:,4], marker='o', color='lightcoral',s=10.0)
plt.xlabel('pmRA',fontsize=14)
plt.ylabel('pmDEC',fontsize=14)

plt.figure(2)
highdataGmag = highdata[:,5]
highdataBPRP = highdata[:,6]-highdata[:,7]
plt.scatter(highdataBPRP, highdataGmag, marker='o', color='lightcoral',s=5.0)
plt.xlabel('BP-RP',fontsize=14)
plt.ylabel('Gmag',fontsize=14)
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向

plt.figure(3)
plt.scatter(waidata[:,0], waidata[:,1], marker='o', color='grey',s=5.0)
plt.scatter(highdata[:,0], highdata[:,1], marker='o', color='lightcoral',s=20.0)
plt.xlabel('RA',fontsize=14)
plt.ylabel('DEC',fontsize=14)