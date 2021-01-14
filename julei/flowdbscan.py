# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 13:51:38 2021

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


data = np.loadtxt('flow2.txt')
print(len(data))



dfdata = pd.DataFrame(data)
dfdata = dfdata.sample(n=20000)
npdfdata = np.array(dfdata)

data_zs = np.copy(npdfdata)

from sklearn.neighbors import NearestNeighbors
nearest_neighbors = NearestNeighbors(n_neighbors=14)
neighbors = nearest_neighbors.fit(data_zs)
distances, indices = neighbors.kneighbors(data_zs)




distances = np.sort(distances[:,13], axis=0)

fig = plt.figure(0)
plt.plot(distances)
plt.xlabel("Points")
plt.ylabel("Distance")


from kneed import KneeLocator
i = np.arange(len(distances))
knee = KneeLocator(i, distances, S=1, curve='concave', direction='increasing', interp_method='polynomial',online=True)
fig = plt.figure(1)
knee.plot_knee()
print(knee.all_knees_y)

plt.xlabel("Points")
plt.ylabel("Distance")

print(distances[knee.knee])
