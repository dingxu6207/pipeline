# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 16:17:45 2021

@author: dingxu
"""

import numpy as np
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import mixture
from matplotlib.pyplot import MultipleLocator
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import imageio
from scipy.stats import norm

np.random.seed(8)

data = np.loadtxt('NGC7142.txt')#6791
print(len(data))
#data = data[data[:,2]>0]
#data = data[data[:,2]<1]
#data[:,2] = data[:,2]
data = data[data[:,3]<15]
data = data[data[:,3]>-15]

data = data[data[:,4]<15]
data = data[data[:,4]>-15]

X = np.copy(data[:,0:5])


def probilitydensity(inputdata):
    pddata = np.copy(inputdata)
    n,bins,patches = plt.hist(pddata, bins=500, density = 1, facecolor='blue', alpha=0.5)
    #bins = np.array(inputdata)
    sigma = np.std(pddata)
    mu = np.average(pddata)
    pdvalue = ((1/(np.power(2*np.pi,0.5)*sigma))*np.exp(-0.5*np.power((bins-mu)/sigma,2)))
    
    return bins,pdvalue
    
bins,pdvaluera = probilitydensity(X[:,2]) 
plt.plot(bins, pdvaluera) 
    
'''
plt.figure(20)
#plt.hist(np.around(X[:,3],3), bins=500, density = 0, facecolor='blue', alpha=0.5)
n,bins,patches = plt.hist(X[:,0], bins=len(data), density = 1, facecolor='blue', alpha=0.5)
sigma = np.std(X[:,0])
mu = np.average(X[:,0])

bf = norm.pdf(bins, mu, sigma)

plt.plot(bins, bf, 'k--', linewidth=2.0)

y=((1/(np.power(2*np.pi,0.5)*sigma))*np.exp(-0.5*np.power((bins-mu)/sigma,2)))
plt.plot(bins,y,color='orange',ls='--',lw=2)
'''
