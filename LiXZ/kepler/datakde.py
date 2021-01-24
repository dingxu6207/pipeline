# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 13:37:14 2020

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

#lightdata = np.loadtxt('savedatasample3.txt') 
lightdata = np.loadtxt('alldatasample35.txt') 


import seaborn as sns
sns.set()


lightdata = lightdata[lightdata[:,100] > 50]
lightdata = lightdata[lightdata[:,101] < 1]
#lightdata = lightdata[lightdata[:,103] < 1.15]
#lightdata = lightdata[lightdata[:,103] > 0.85]
newdata = lightdata[lightdata[:,101] < 0.5]
d4data = lightdata[lightdata[:,101] > 0.5]

dfdata = pd.DataFrame(newdata)
dfdata = dfdata.sample(n=70590)
npdfdata = np.array(dfdata)

df4data = pd.DataFrame(d4data)
df4data = df4data.sample(n=50000)
np4dfdata = np.array(df4data)

alldata = np.row_stack((np4dfdata, npdfdata))
lightdata = np.copy(alldata)

'''
lightdata = lightdata[lightdata[:,103]<1.04]
lightdata = lightdata[lightdata[:,103]>0.92]
'''
print(len(np4dfdata))
print(len(npdfdata))

plt.figure(0)
incldata = lightdata[:,100]
sns.kdeplot(incldata,shade=True)
plt.title('incl')

plt.figure(1)
qdata = lightdata[:,101]
sns.kdeplot(qdata,shade=True)
plt.title('q')

plt.figure(2)
rdata = lightdata[:,102]
sns.kdeplot(rdata,shade=True)
plt.title('r')

plt.figure(3)
tdata = lightdata[:,103]
sns.kdeplot(tdata,shade=True)
plt.title('T2/T1')

np.savetxt('alldata35.txt', alldata)




