# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 21:16:11 2020

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


k =2    #聚类的级别
iteration = 500    #剧烈最大循环次数
 
data = pd.DataFrame(X)
data_zs = 1.0 *(data-data.mean())/data.std()    #数据标准化
 
model = KMeans(n_clusters=k, n_jobs=4, max_iter=iteration)    #分为k类，并发数4
model.fit(data_zs)    #开始聚类，训练模型
 
#简单打印结果
r1 = pd.Series(model.labels_).value_counts()    #统计各个类别的数目
r2 = pd.DataFrame(model.cluster_centers_)    #找出聚类中心
r = pd.concat([r2, r1], axis=1)    #横向连接（0是纵向），得到聚类中心对应的类别下的数目
r.columns = list(data.columns) + [u'类别数目']    #重命名表头
#print(r)
 
#详细输出原始数据及其类别
 
r = pd.concat([data, pd.Series(model.labels_, index=data.index)], axis=1)    #详细输出每个样本对应的类别
r.columns = list(data.columns) + [u'聚类类别']    #重命名表头

 
tsne = TSNE()
tsne.fit_transform(data_zs)    #进行降维
tsne = pd.DataFrame(tsne.embedding_, index=data_zs.index)    #转换数据格式
 

 
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
 
#不同类别用不同颜色和样式绘图
 
d = tsne[r[u'聚类类别']==0]
plt.plot(d[0], d[1], 'r.')
 
d = tsne[r[u'聚类类别']==1]
plt.plot(d[0], d[1], 'go')
 

plt.show()


