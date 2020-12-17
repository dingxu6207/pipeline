# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 18:10:27 2020

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
import hdbscan

np.random.seed(8)

data = np.loadtxt('NGC6791.txt')
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

#clt = DBSCAN(eps = 0.12, min_samples = 11)
#datalables = clt.fit_predict(data_zs)


clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
datalables = clusterer.fit_predict(data_zs)

r1 = pd.Series(datalables).value_counts()
print(r1)



