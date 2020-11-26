# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 22:13:14 2020

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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import imageio
from sklearn.cluster import SpectralClustering
from sklearn import metrics 
from sklearn.cluster import Birch

np.random.seed(8)

data = np.loadtxt('Gaiadata.txt')
print(len(data))
#data = data[data[:,2]>0]
#data = data[data[:,2]<1]

data = data[data[:,3]<15]
data = data[data[:,3]>-15]

data = data[data[:,4]<15]
data = data[data[:,4]>-15]

X = np.copy(data[:,0:5])


X = StandardScaler().fit_transform(X)
data_zs = np.copy(X)


y_pred = Birch(n_clusters = 2,threshold = 0.8).fit_predict(X)

r1 = pd.Series(y_pred).value_counts()
print(r1)

datapro = np.column_stack((data ,y_pred))

highdata = datapro[datapro[:,8] == 1]
lowdata = datapro[datapro[:,8] == 0]

plt.figure(1)
plt.scatter(highdata[:,3], highdata[:,4], marker='o', color='lightcoral',s=5.0)
plt.scatter(lowdata[:,3], lowdata[:,4], marker='o', color='grey',s=5.0)
plt.xlabel('pmRA',fontsize=14)
plt.ylabel('pmDEC',fontsize=14)
plt.xlim((-15,15))
plt.ylim((-15,15))

plt.figure(2)
hparallax = highdata[:,2]
hGmag = highdata[:,5]
lparallax = lowdata[:,2]
lGmag = lowdata[:,5]
plt.scatter(lGmag, lparallax, marker='o', color='grey',s=5)
plt.scatter(hGmag, hparallax, marker='o', color='lightcoral',s=5)
plt.xlabel('Gmag',fontsize=14)
plt.ylabel('parallax',fontsize=14)



