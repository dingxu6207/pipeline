# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 21:20:30 2021

@author: dingxu
"""

import pandas as pd
import numpy as np
from PyAstronomy.pyasl import foldAt
from PyAstronomy.pyTiming import pyPDM
import matplotlib.pylab as plt
from scipy import interpolate

CSV_FILE_PATH = '55.csv'
dfdata = pd.read_csv(CSV_FILE_PATH)

hjd = dfdata['HJD']
mag = dfdata['mag']

nphjd = np.array(hjd)
npmag = np.array(mag)

hang = 287
nphjd = nphjd[0:hang]
npmag = npmag[0:hang]-np.mean(npmag[0:hang])

phases = foldAt(nphjd, 0.3702918)
sortIndi = np.argsort(phases)
# ... and, second, rearrange the arrays.
phases = phases[sortIndi]
resultmag = npmag[sortIndi]

#plt.plot(phases, resultmag,'.')

listmag = resultmag.tolist()
listmag.extend(listmag)

listphrase = phases.tolist()
listphrase.extend(listphrase+np.max(listphrase)) 

indexmag = listmag.index(max(listmag))

nplistphrase = np.array(listphrase)
nplistphrase = nplistphrase-nplistphrase[indexmag]
nplistmag = np.array(listmag)

#phasemag = np.concatenate([nplistphrase, nplistmag],axis=1)


phasemag = np.vstack((nplistphrase, nplistmag)) #纵向合并矩阵
phasemag = phasemag.T

phasemag = phasemag[phasemag[:,0]>0]
phasemag = phasemag[phasemag[:,0]<1]


phrase = phasemag[:,0]
flux = phasemag[:,1]
sx1 = np.linspace(0,1,200)
func1 = interpolate.UnivariateSpline(phrase, flux,s=0.37)#强制通过所有点
sy1 = func1(sx1)


plt.figure(1)
plt.plot(sx1, sy1,'.', c='r')#对原始数据画散点图

plt.figure(0)
plt.plot(phrase, flux,'.')
plt.plot(sx1, sy1,'.', c='r')#对原始数据画散点图
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
plt.xlabel('Phrase',fontsize=14)
plt.ylabel('mag',fontsize=14)

interdata = np.vstack((sx1,sy1))
np.savetxt('ztf1.txt', interdata.T)


