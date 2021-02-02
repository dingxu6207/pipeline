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
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import imageio

np.random.seed(8)

data = np.loadtxt('NGC7142.txt')
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


from sklearn.neighbors import NearestNeighbors
nearest_neighbors = NearestNeighbors(n_neighbors=11)
neighbors = nearest_neighbors.fit(X)
distances, indices = neighbors.kneighbors(X)

tempdistance = distances

tempones = np.ones(11,dtype = np.uint)

sumtemp = np.dot(tempdistance, tempones)


distances = np.sort(distances[:,1], axis=0)
#distances = np.sort(sumtemp/10, axis=0)
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

plt.text(13092, 0.56, 'eps=0.6', color = "b", style = "italic", weight = "light", verticalalignment='center', horizontalalignment='left',rotation=0) #给散点加标签
