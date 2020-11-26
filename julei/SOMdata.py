# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 23:13:02 2020

@author: dingxu
"""

from minisom import MiniSom
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import imageio
from mpl_toolkits.mplot3d import Axes3D

#https://github.com/JustGlowing/minisom/blob/master/examples/Clustering.ipynb

data = np.loadtxt('Gaiadata.txt')

data = data[data[:,3]<15]
data = data[data[:,3]>-15]

data = data[data[:,4]<15]
datay = data[data[:,4]>-15]

X = np.copy(datay[:,0:5])
X = StandardScaler().fit_transform(X)
data_zs = np.copy(X)
data = data_zs


# Initialization and training
som_shape = (1, 2)
som = MiniSom(som_shape[0], som_shape[1], data.shape[1], sigma=0.3, learning_rate=0.00001,
              neighborhood_function='gaussian', random_seed=10)

som.train_batch(data, 500000, verbose=True)

# each neuron represents a cluster
winner_coordinates = np.array([som.winner(x) for x in data]).T
# with np.ravel_multi_index we convert the bidimensional
# coordinates to a monodimensional index
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

r1 = pd.Series(cluster_index).value_counts()
print(r1)

datapro = np.column_stack((datay ,cluster_index))

highdata = datapro[datapro[:,8] == 0]
lowdata = datapro[datapro[:,8] == 1]

plt.figure(1)
plt.scatter(highdata[:,3], highdata[:,4], marker='o', color='lightcoral',s=5.0)
plt.scatter(lowdata[:,3], lowdata[:,4], marker='o', color='grey',s=5.0)
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

'''
dataable = np.column_stack((data_zs ,cluster_index))
pddata = pd.DataFrame(dataable)
datazs = pd.DataFrame(data_zs)

tsne = TSNE(n_components=3, learning_rate=100, n_iter=1000, init='pca')
tsne.fit_transform(data_zs)    #进行降维
tsne = pd.DataFrame(tsne.embedding_, index=datazs.index)    #转换数据格式


plt.figure(3)
ax1 = plt.axes(projection='3d')


gif_images = []
for t in range (0,1000):
    if t == 360:
        break
    plt.cla() # 此命令是每次清空画布，所以就不会有前序的效果
    ax1.set_title('SOM')
   
    d = tsne[pddata.iloc[:,5] == 1]
    ax1.scatter3D(d[0], d[1], d[2], c ='b', marker='o', s=1)
    plt.pause(0.01)
    
    d = tsne[pddata.iloc[:,5] == 0]
    ax1.scatter3D(d[0], d[1], d[2], c ='r', marker='x', s=10)
      
    plt.pause(0.01)
    plt.savefig('1.jpg')
    
    ax1.view_init(elev=0., azim=t+1)
    gif_images.append(imageio.imread('1.jpg'))
    
    
imageio.mimsave("SOMtest.gif",gif_images,fps=20)
'''