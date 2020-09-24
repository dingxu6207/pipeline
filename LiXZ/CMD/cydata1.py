# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 09:27:03 2020

@author: dingxu
"""

#RA_ICRS;DE_ICRS;Plx;pmRA;pmDE;Gmag;BPmag;RPmag


import numpy as np
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import mixture



#np.random.seed(7)

data = np.loadtxt('cydata1.txt')



X = np.copy(data[:,0:5])




ZSX = np.copy(X)
clst = mixture.GaussianMixture(n_components = 2)
predicted_lables = clst.fit_predict(ZSX)

r1 = pd.Series(predicted_lables).value_counts()

print(r1)


Gmag = data[:,5]
BPRP = data[:,6]-data[:,7] 

plt.figure(0)
plt.plot(BPRP, Gmag, '.')
plt.xlabel('BP-RP',fontsize=14)
plt.ylabel('Gmag',fontsize=14)
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向


datalable = np.column_stack((data ,predicted_lables))

cydata = datalable[datalable[:,8]==1]
cyGmag = cydata[:,5]
cyBPRP = cydata[:,6]-cydata[:,7] 
plt.figure(1)
plt.scatter(cyBPRP, cyGmag, marker='o', color='lightcoral',s=10.0)
plt.xlabel('BP-RP',fontsize=14)
plt.ylabel('Gmag',fontsize=14)
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
np.savetxt('cydata1.txt', cydata[:,0:8])


waidata = datalable[datalable[:,8]==0]
waiGmag = waidata[:,5]
waiBPRP = waidata[:,6]-waidata[:,7] 
plt.figure(2)
plt.scatter(waiBPRP, waiGmag, marker='o', color='grey',s=10.0)
plt.xlabel('BP-RP',fontsize=14)
plt.ylabel('Gmag',fontsize=14)
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向





plt.figure(3)
plt.scatter(waidata[:,3], waidata[:,4], marker='o', color='grey',s=10.0)
plt.scatter(cydata[:,3], cydata[:,4], marker='o', color='lightcoral',s=10.0)
plt.xlabel('pmRA',fontsize=14)
plt.ylabel('pmDEC',fontsize=14)


#plt.figure(4)
#plt.plot(cydata[:,3], cydata[:,4], '.', color='lightcoral')
