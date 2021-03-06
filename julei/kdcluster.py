# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:11:50 2020

@author: dingxu
"""

import numpy as np
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import mixture
from matplotlib.pyplot import MultipleLocator
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import imageio

np.random.seed(8)

data = np.loadtxt('NGC6397.txt')#6791
print(len(data))
#data = data[data[:,2]>0]
#data = data[data[:,2]<1]
#data[:,2] = data[:,2]
data = data[data[:,3]<15]
data = data[data[:,3]>-15]

data = data[data[:,4]<15]
data = data[data[:,4]>-15]

X = np.copy(data[:,0:5])

plt.figure(20)
#plt.hist(np.around(X[:,3],3), bins=500, density = 0, facecolor='blue', alpha=0.5)
plt.hist(X[:,0], bins=500, density = 0, facecolor='blue', alpha=0.5)


X = StandardScaler().fit_transform(X)
#X = MinMaxScaler().fit_transform(X)
data_zs = np.copy(X)

plt.figure(21)
#plt.hist(np.around(data_zs[:,3],3), bins=500, density = 0, facecolor='blue', alpha=0.5)
plt.hist(data_zs[:,3], bins=500, density = 0, facecolor='blue', alpha=0.5)


plt.figure(22)
#plt.hist(np.around(data_zs[:,3],3), bins=500, density = 0, facecolor='blue', alpha=0.5)
plt.hist2d(data_zs[:,3],data_zs[:,4], bins=500, density = 0, facecolor='blue', alpha=0.5)


from sklearn.neighbors import NearestNeighbors
nearest_neighbors = NearestNeighbors(n_neighbors=11)
neighbors = nearest_neighbors.fit(X)
distances, indices = neighbors.kneighbors(X)

tempdistance = distances

tempones = np.ones(11,dtype = np.uint)

sumtemp = np.dot(tempdistance, tempones)


distances = np.sort(distances[:,10], axis=0)
#distances = np.sort(sumtemp/10, axis=0)
plt.plot(0)
fig = plt.figure(figsize=(5, 5))
plt.plot(distances)
plt.xlabel("Points")
plt.ylabel("Distance")
plt.savefig("Distance_curve.png", dpi=300)

from kneed import KneeLocator
i = np.arange(len(distances))
knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial',online=True)
fig = plt.figure(figsize=(5, 5))
knee.plot_knee()
print(knee.all_knees_y)

plt.xlabel("Points")
plt.ylabel("Distance")

print(distances[knee.knee])

plt.text(13279, 0.90, 'eps=0.97', color = "b", style = "italic", weight = "light", verticalalignment='center', horizontalalignment='left',rotation=0) #给散点加标签


plt.figure(10)
plt.hist(np.around(distances,3), bins=1000, density = 0, facecolor='blue', alpha=0.5)
plt.xlabel("Distance")
plt.ylabel("Points")
