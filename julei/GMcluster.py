# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 22:04:49 2020

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

data = data[data[:,3]<15]
data = data[data[:,3]>-15]

data = data[data[:,4]<15]
datay = data[data[:,4]>-15]

#data = data[data[:,7]<18]

X = np.copy(datay[:,0:5])

X = StandardScaler().fit_transform(X)
data = pd.DataFrame(X)
#data_zs = 1.0 *(data-data.mean())/data.std()    #数据标准化
data_zs = data

#print(np.sum(data_zs-data_zsd))

clst = mixture.GaussianMixture(n_components = 2)

predicted_lables = clst.fit_predict(data_zs)

prodata = clst.predict_proba(data_zs)

r1 = pd.Series(predicted_lables).value_counts()

print(r1)

datalable = np.column_stack((data ,predicted_lables))
#datalable = datalable[np.argsort(datalable[:,5])]

r = pd.DataFrame(datalable)
r.columns = list(data.columns) + [u'聚类类别'] 
r.to_excel("1.xls")

tsne = TSNE()
tsne.fit_transform(data_zs)    #进行降维
tsne = pd.DataFrame(tsne.embedding_, index=data_zs.index)    #转换数据格式
 

plt.figure(0)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
 
#不同类别用不同颜色和样式绘图
 
d = tsne[r[u'聚类类别']==0]
plt.plot(d[0], d[1], 'r.')
 
d = tsne[r[u'聚类类别']==1]
plt.plot(d[0], d[1], 'b.')
 

datapro = np.column_stack((datay ,prodata))
highdata = datapro[datapro[:,9] > datapro[:,8]]
highdata = highdata[highdata[:,9]>0.8]
print(len(highdata))
lowdata = datapro[datapro[:,9] < datapro[:,8]]
waidata = lowdata

plt.figure(1)
plt.scatter(waidata[:,3], waidata[:,4], marker='o', color='red',s=5.0)
plt.scatter(highdata[:,3], highdata[:,4], marker='o', color='blue',s=5.0)
plt.xlabel('pmRA',fontsize=14)
plt.ylabel('pmDEC',fontsize=14)
plt.xlim((-15,15))
plt.ylim((-15,15))

plt.figure(2)
datazero = r[r[u'聚类类别']==0]
datazero = np.array(datazero)

dataone = r[r[u'聚类类别']==1]
dataone = np.array(dataone)

plt.scatter(datazero[:,3],datazero[:,4], marker='o', color='r',s=5.0)
plt.scatter(dataone[:,3],dataone[:,4], marker='o', color='blue',s=5.0)