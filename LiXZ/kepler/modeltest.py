# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 23:16:20 2020

@author: dingxu
"""

from tensorflow.keras.models import load_model
#from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing    
model = load_model('phoebemodel.h5') #phoebemodel.h5
#model = load_model('phmod.h5')
#model = load_model('m3.h5')
model.summary()

#path = 'E:\\shunbianyuan\\data\\kepler\\KIC_name\\'
#file = 'KIC 4036687.txt'

path = 'E:\\shunbianyuan\\data\\kpdata\\code\\'
file = '3207.lc'

data = np.loadtxt(path+file)

#datay = 10**(data[:,1]/(-2.5))
datay = data[:,1]
datay = -2.5*np.log10(datay)
datay = datay-np.mean(datay)

plt.figure(0)
plt.plot(data[:,0], datay, '.')
#plt.plot(data[:,0], data, '.')




nparraydata = np.reshape(datay,(1,100))

prenpdata = model.predict(nparraydata)

print(prenpdata)
