# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:01:23 2020

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
import imageio
np.random.seed(8)

data = np.loadtxt('Gaiadata.txt')

data = data[data[:,3]<15]
data = data[data[:,3]>-15]

data = data[data[:,4]<15]
datay = data[data[:,4]>-15]

X = np.copy(datay[:,0:5])
X = StandardScaler().fit_transform(X)
data_zs = np.copy(X)

clst = mixture.GaussianMixture(n_components = 2)
predicted_lables = clst.fit_predict(data_zs)
prodata = clst.predict_proba(data_zs)




proflag = []
for i in range (0,len(data_zs)):
    if prodata[i,1]>0.985:
        proflag.append(1)
    else:
        proflag.append(0)

predicted_lables = np.array(proflag)

r1 = pd.Series(predicted_lables).value_counts()
print(r1)

datapro = np.column_stack((datay ,predicted_lables))

highdata = datapro[datapro[:,8] == 1]
lowdata = datapro[datapro[:,8] == 0]

plt.figure(1)
plt.scatter(lowdata[:,3], lowdata[:,4], marker='o', color='grey',s=5.0)
plt.scatter(highdata[:,3], highdata[:,4], marker='o', color='lightcoral',s=5.0)
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


dataable = np.column_stack((data_zs ,predicted_lables))
pddata = pd.DataFrame(dataable)
datazs = pd.DataFrame(data_zs)

tsne = TSNE(n_components=3, learning_rate=100, n_iter=1000, init='pca')
tsne.fit_transform(data_zs)    #进行降维
tsne = pd.DataFrame(tsne.embedding_, index=datazs.index)    #转换数据格式


plt.figure(3)
ax1 = plt.axes(projection='3d')

#plt.rcParams['font.sans-serif'] = ['SimHei']
#plt.rcParams['axes.unicode_minus'] = False
#不同类别用不同颜色和样式绘图
#plt.plot(d[0], d[1], 'b.')
#plt.plot(d[0], d[1], 'r.')

gif_images = []
for t in range (0,1000):
    if t == 360:
        break
    plt.cla() # 此命令是每次清空画布，所以就不会有前序的效果
    ax1.set_title('GMM')
   
    d = tsne[pddata.iloc[:,5] == 0]
    ax1.scatter3D(d[0], d[1], d[2], c ='b', marker='o', s=1)
    plt.pause(0.01)
    
    d = tsne[pddata.iloc[:,5] == 1]
    ax1.scatter3D(d[0], d[1], d[2], c ='r', marker='x', s=10)
      
    plt.pause(0.01)
    plt.savefig('1.jpg')
    
    ax1.view_init(elev=0., azim=t+1)
    gif_images.append(imageio.imread('1.jpg'))
    
    
imageio.mimsave("GMMtest.gif",gif_images,fps=20)


'''
fig = plt.figure(3)
ax1 = plt.axes(projection='3d')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


d = tsne[pddata.iloc[:,5] == 1]
#plt.plot(d[0], d[1], 'r.')
ax1.scatter3D(d[0], d[1], d[2], c ='r', marker='x', s=10)

d = tsne[pddata.iloc[:,5] == 0]
ax1.scatter3D(d[0], d[1], d[2], c = 'b', marker='o', s=1)

 
fig = plt.figure(4)
ax2 = plt.axes(projection='3d')
d = tsne[pddata.iloc[:,5] == 0]
#plt.plot(d[0], d[1],'b.')
ax2.scatter3D(d[0], d[1], d[2], c = 'b', marker='o')


fig = plt.figure(5)
ax3 = plt.axes(projection='3d')
d = tsne[pddata.iloc[:,5] == 1]
#plt.plot(d[0], d[1],'b.')
ax3.scatter3D(d[0], d[1], d[2], c = 'r', marker='o')
'''