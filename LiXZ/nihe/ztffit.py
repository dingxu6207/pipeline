# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 17:28:30 2021

@author: dingxu
"""

#Period: 0.3702918 ID: ZTFJ000006.67+641227.6 SourceID: 55

import pandas as pd
import numpy as np
from PyAstronomy.pyasl import foldAt
from PyAstronomy.pyTiming import pyPDM
import matplotlib.pylab as plt

CSV_FILE_PATH = '55.csv'
dfdata = pd.read_csv(CSV_FILE_PATH)

hjd = dfdata['HJD']
mag = dfdata['mag']

nphjd = np.array(hjd)
npmag = np.array(mag)
#nphjd = nphjd[0:287]
npmag1 = npmag[0:287]-np.mean(npmag[0:287])
npmag2 = npmag[287:]-np.mean(npmag[287:])

npmag = np.concatenate([npmag1,npmag2],axis=0)
#npmag = np.row_stack((npmag1, npmag2))

phases = foldAt(nphjd, 0.3702918)
sortIndi = np.argsort(phases)
# ... and, second, rearrange the arrays.
phases = phases[sortIndi]
resultmag = npmag[sortIndi]

listmag = resultmag.tolist()
listmag.extend(listmag)

listphrase = phases.tolist()
listphrase.extend(listphrase+np.max(listphrase)) 

indexmag = listmag.index(max(listmag))


nplistphrase = np.array(listphrase)
nplistphrase = nplistphrase-nplistphrase[indexmag]
nplistmag = np.array(listmag)



duanx = nplistphrase[188:780]
duany = nplistmag[188:780]
plt.figure(0)
#plt.plot(phases, resultmag, '.')

plt.plot(duanx, duany, '.')
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向


a=np.polyfit(duanx,duany,17)#用2次多项式拟合x，y数组
b=np.poly1d(a)#拟合完之后用这个函数来生成多项式对象
c=b(duanx)#生成多项式对象之后，就是获取x在这个多项式处的值
plt.figure(1)
plt.plot(duanx,duany,'.')#对原始数据画散点图
plt.plot(duanx,c,ls='--',c='red')#对拟合之后的数据，也就是x，c数组画图
plt.legend()
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
plt.xlabel('Phrase',fontsize=14)
plt.ylabel('mag',fontsize=14)
phrasefluxdata = np.vstack((duanx, c))
np.savetxt('lightcurve.txt', phrasefluxdata.T)
