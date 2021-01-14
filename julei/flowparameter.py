# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 13:53:00 2021

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

data = np.loadtxt('wanflow.txt')
print(len(data))

X = np.copy(data)


#X = StandardScaler().fit_transform(X)
data_zs = np.copy(X)


res = []
i = 0
for eps in np.arange(0.05,2,0.01):
    # 迭代不同的min_samples值
    for min_samples in range(2,15):
        clt = DBSCAN(eps = eps, min_samples = min_samples)
        datalables = clt.fit_predict(data_zs)

        n_clusters = len([i for i in set(datalables)])

        stats = str(pd.Series([i for i in datalables]).value_counts().values)
        
        res.append({'eps':eps,'min_samples':min_samples,'n_clusters':n_clusters,'stats':stats})
        
        i = i+1
        print(str(i)+' '+'it is ok!')

# 将迭代后的结果存储到数据框中        
df = pd.DataFrame(res)

# 根据条件筛选合理的参数组合
df2cluster = df.loc[df.n_clusters == 2, :]

df3cluster = df.loc[df.n_clusters == 3, :]

df4cluster = df.loc[df.n_clusters == 4, :]