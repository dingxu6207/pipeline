# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 14:49:40 2021

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


#X = StandardScaler().fit_transform(X)
data_zs = np.copy(npdfdata)

clt = DBSCAN(eps = 0.52, min_samples = 14)
datalables = clt.fit_predict(data_zs)

r1 = pd.Series(datalables).value_counts()

print(r1)



datapro = np.column_stack((data_zs ,datalables))

highdata = datapro[datapro[:,2] == 0]
lowdata = datapro[datapro[:,2] == -1]

np.savetxt('highdata.txt', highdata)
np.savetxt('lowdata.txt',  lowdata)

plt.figure(1)
plt.scatter(lowdata[:,0], lowdata[:,1], marker='o', color='grey',s=5.0)
plt.scatter(highdata[:,0], highdata[:,1], marker='o', color='lightcoral',s=5.0)


import seaborn as sns
sns.set()
sns.kdeplot(lowdata[:,1],shade=True)
sns.kdeplot(highdata[:,1],shade=True)