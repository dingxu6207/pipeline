# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 23:52:14 2020

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

bemeberdata = np.loadtxt('Be99member.txt')
hang = len(bemeberdata)

data = np.loadtxt('Be99.txt')
print(len(data))
#data = data[data[:,2]>0]
#data = data[data[:,2]<1]

#data = data[data[:,3]<150]
#data = data[data[:,3]>-150]

#data = data[data[:,4]<150]
#data = data[data[:,4]>-150]

X = np.copy(data[:,0:5])


X = StandardScaler().fit_transform(X)
data_zs = np.copy(X)

clt = DBSCAN(eps = 0.22, min_samples = 12)
datalables = clt.fit_predict(data_zs)

r1 = pd.Series(datalables).value_counts()

print(r1)

datapro = np.column_stack((data ,datalables))

highdata = datapro[datapro[:,8] == 0]
lowdata = datapro[datapro[:,8] == -1]

plt.figure(0)
plt.scatter(lowdata[:,0], lowdata[:,1], marker='o', color='grey',s=5.0)
plt.scatter(highdata[:,0], highdata[:,1], marker='o', color='lightcoral',s=5.0)
plt.xlabel('RA',fontsize=14)
plt.ylabel('DEC',fontsize=14)
for i in range(10):
    ra = bemeberdata[i,1]
    dec = bemeberdata[i,2]
    #plt.scatter(ra, dec, marker='o', color='darkgreen', s=1)
    plt.text(ra, dec, 'V'+str(i+17), fontsize=10, color = "b", style = "italic", weight = "light", verticalalignment='center', horizontalalignment='left',rotation=0) #给散点加标签

#plt.text(350.38229167, 71.76819444, 'V'+str(i+30), fontsize=10, color = "b", style = "italic", weight = "light", verticalalignment='center', horizontalalignment='left',rotation=0) #给散点加标签
#plt.text(350.26229167, 71.82222222, 'V'+str(i+30), fontsize=10, color = "b", style = "italic", weight = "light", verticalalignment='center', horizontalalignment='left',rotation=0) #给散点加标签

plt.figure(1)
plt.scatter(lowdata[:,3], lowdata[:,4], marker='o', color='grey',s=5.0)
plt.scatter(highdata[:,3], highdata[:,4], marker='o', color='lightcoral',s=5.0)
pmdata = np.vstack((highdata[:,3], highdata[:,4]))
np.savetxt('PM.txt', pmdata)
plt.xlabel('pmRA',fontsize=14)
plt.ylabel('pmDEC',fontsize=14)
plt.xlim((-15,15))
plt.ylim((-15,15))
for i in range(10):
    pmra = bemeberdata[i,4]
    pmdec = bemeberdata[i,5]
    plt.scatter(pmra, pmdec, marker='o', color='darkgreen', s=1)
    if i == 1:
        plt.text(pmra, pmdec, 'V'+str(i+17), fontsize=10, color = "b", style = "italic", weight = "light", verticalalignment='center', horizontalalignment='left',rotation=0) #给散点加标签



plt.figure(2)
hparallax = highdata[:,2]
hGmag = highdata[:,5]

lparallax = lowdata[:,2]
lGmag = lowdata[:,5]

plt.scatter(lGmag, lparallax, marker='o', color='grey',s=5)
plt.scatter(hGmag, hparallax, marker='o', color='lightcoral',s=5)
pallaxdata = np.vstack((lGmag, lparallax))
np.savetxt('parallax.txt', pallaxdata)
plt.xlabel('Gmag',fontsize=14)
plt.ylabel('parallax',fontsize=14)
for i in range(10):
    mparallax = bemeberdata[i,3]
    mGmag = bemeberdata[i,6]
    plt.scatter(mGmag, mparallax, marker='o', color='darkgreen', s=20)
    plt.text(mGmag, mparallax, 'V'+str(i+17), fontsize=10, color = "b", style = "italic", weight = "light", verticalalignment='center', horizontalalignment='left',rotation=0) #给散点加标签




plt.figure(3)
highdataGmag = highdata[:,5]
highdataBPRP = highdata[:,6]-highdata[:,7]
loaddata = np.vstack((highdataGmag,highdataBPRP))
np.savetxt('BPRPG.txt', loaddata)
plt.xlim((-1,4))
plt.ylim((10,22))
plt.scatter((lowdata[:,6]-lowdata[:,7]), lowdata[:,5], marker='o', color='grey',s=5)
plt.scatter(highdataBPRP, highdataGmag, marker='o', color='lightcoral',s=5)
x_major_locator = MultipleLocator(1)
plt.xlabel('BP-RP',fontsize=14)
#plt.xlabel('G-RP',fontsize=14)
plt.ylabel('Gmag',fontsize=14)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向

for i in range(10):
    G = bemeberdata[i,6]
    BPRP = bemeberdata[i,7]-bemeberdata[i,8]
    plt.scatter(BPRP, G, marker='o', color='darkgreen', s=20)
    plt.text(BPRP, G, 'V'+str(i+17), fontsize=10, color = "b", style = "italic", weight = "light", verticalalignment='center', horizontalalignment='left',rotation=0) #给散点加标签


plt.figure(5)

ax1 = plt.axes(projection='3d')
ax1.scatter3D(lowdata[:,0], lowdata[:,1], lowdata[:,2], c = 'b', marker='o', s=0.01)
ax1.scatter3D(highdata[:,0], highdata[:,1], highdata[:,2], c ='r', marker='o', s=1)
ax1.set_xlabel('RA')
#ax1.set_xlim(-6, 4)  #拉开坐标轴范围显示投影
ax1.set_ylabel('DEC')
#ax1.set_ylim(-4, 6)
ax1.set_zlabel('Parallax')
#ax1.set_zlim(-2, 2)
ax1.set_title('Be99')

#ax1.view_init(elev=30, azim=30)


