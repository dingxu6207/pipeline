# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 22:40:19 2020

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
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
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

'''
clt = DBSCAN(eps = 0.1, min_samples = 4)
datalables = clt.fit_predict(data_zs)

r1 = pd.Series(datalables).value_counts()

print(r1)
'''




res = []
i = 0
for eps in np.arange(0.1,0.5,0.01):
    # 迭代不同的min_samples值
    for min_samples in range(2,15):
        clt = DBSCAN(eps = eps, min_samples = min_samples)
        datalables = clt.fit_predict(data_zs)
        
        try:
            datapro = np.column_stack((data_zs ,datalables))
            highdata = datapro[datapro[:,5] == 0]
            nearest_neighbors = NearestNeighbors(n_neighbors=3)
            neighbors = nearest_neighbors.fit(highdata[0:5])
            distances, indices = neighbors.kneighbors(highdata[0:5])
            hang,lie = distances.shape
            sscore = np.sum(distances[:,1])/hang
            
            print(sscore)
            
        except:
            sscore = -1

        n_clusters = len([i for i in set(datalables)])

        stats = str(pd.Series([i for i in datalables]).value_counts().values)
        
        res.append({'eps':eps,'min_samples':min_samples,'n_clusters':n_clusters,'score':sscore,'stats':stats})
        
        i = i+1
        print(str(i)+' '+'it is ok!')

# 将迭代后的结果存储到数据框中        
df = pd.DataFrame(res)

# 根据条件筛选合理的参数组合
df2cluster = df.loc[df.n_clusters == 2, :]

epsdata = df2cluster['eps']
mindata = df2cluster['min_samples']

'''
plt.figure(0)
plt.plot(epsdata, mindata, '*')

plt.figure(1)
plt.plot(mindata, epsdata, '*')
'''