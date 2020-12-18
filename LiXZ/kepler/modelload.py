# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:16:02 2020

@author: dingxu
"""

from tensorflow.keras.models import load_model
#from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing    
#model = load_model('phoebemodel.h5') #phoebemodel.h5
#model = load_model('phmod.h5') #phoebemodel.h5
#model = load_model('phmodsample2x.h5')
model = load_model('weights-improvement-14563-0.0075.hdf5')
model.summary()

path = 'E:\\shunbianyuan\\data\\kepler\\KIC_name\\'
file = 'KIC 9453192.txt'

data = np.loadtxt(path+file)

#datay = 10**(data[:,1]/(-2.5))
#datay = (datay-np.min(datay))/(np.max(datay)-np.min(datay))
datay = data[:,1]-np.mean(data[:,1])

plt.figure(0)
plt.plot(data[:,0], datay, '.')



plt.figure(1)
hang = data[:,0]*100
inthang = np.uint(hang)
plt.plot(inthang, datay, '.')

npdata = np.vstack((inthang, datay))

lendata = len(npdata.T)

temp = [0 for i in range(100)]
for i in range(lendata):
    index = np.uint(npdata[0,:][i])
    temp[index] = npdata[1,:][i]


        
plt.figure(2)
plt.plot(temp, '.')


nparraydata = np.array(temp)
nparraydata = np.reshape(nparraydata,(1,100))

prenpdata = model.predict(nparraydata)

print(prenpdata)
