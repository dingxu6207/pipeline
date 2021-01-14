# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 01:19:28 2020

@author: dingxu
"""
  
from astropy.coordinates import SkyCoord 
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
     
#VSX的星表数据           
#VSXdata = pd.read_csv('NGC559.csv')
VSXdata = pd.read_csv('NGC7142.csv')

dataradec = VSXdata['Coords']


listdata = dataradec.tolist()

#intlist = int(listdata[0][:2])
print(listdata[0])

RA = '21h45m12.81s'  #21:50:56.794   21 45 10.0 +65 46 18
DEC = '+65d47m02.9s' #21 45 10.77 +65 44 41.3

#RA = listdata[0][0:2]+'h'+ listdata[0][3:5] +'m'+listdata[0][6:11]+'s'
#DEC = listdata[0][13:15]+'d'+ listdata[0][16:18] +'m'+listdata[0][19:23]+'s'

c3 = SkyCoord(RA, DEC, frame='icrs')

print('c3.dec.degree=', c3.dec.degree)
print('c3.ra.degree=', c3.ra.degree)

radectemp = []
for i in range(len(listdata)):
    RA = listdata[i][0:2]+'h'+ listdata[i][3:5] +'m'+listdata[i][6:11]+'s'
    DEC = listdata[i][13:15]+'d'+ listdata[i][16:18] +'m'+listdata[i][19:23]+'s'
    c3 = SkyCoord(RA, DEC, frame='icrs')
    #print('c3.dec.degree=', c3.dec.degree)
    #print('c3.ra.degree=', c3.ra.degree)
    radectemp.append(c3.ra.degree)
    radectemp.append(c3.dec.degree)
    
radec = np.float32(radectemp).reshape(-1,2)

np.savetxt('NGC7142radec.txt', radec)

#GAIA的星表数据
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

clt = DBSCAN(eps = 0.23, min_samples = 14)
datalables = clt.fit_predict(data_zs)

r1 = pd.Series(datalables).value_counts()

print(r1)

datapro = np.column_stack((data ,datalables))

highdata = datapro[datapro[:,8] == 0]
lowdata = datapro[datapro[:,8] == -1]


plt.figure(0)
#plt.scatter(lowdata[:,0], lowdata[:,1], marker='o', color='grey',s=1.0)
plt.scatter(highdata[:,0], highdata[:,1], marker='o', color='lightcoral',s=15.0)
plt.xlabel('RA',fontsize=14)
plt.ylabel('DEC',fontsize=14)
for i in range(len(listdata)):
    ra = radec[i,0]
    dec = radec[i,1]
    plt.scatter(ra, dec, marker='o', color='darkgreen', s=1)
    plt.text(ra, dec, 'V'+str(i+1), fontsize=8, color = "b", style = "italic", weight = "light", verticalalignment='center', horizontalalignment='left',rotation=0) #给散点加标签



 
    
