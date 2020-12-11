# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 11:09:39 2020

@author: dingxu
"""

from tensorflow.keras.models import load_model
#from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing 
   
data = np.loadtxt('savedatasample2.txt')
print(len(data))

for i in range(len(data)):
    data[i,0:100] = -2.5*np.log10(data[i,0:100]) 
    data[i,0:100] = (data[i,0:100])-np.mean(data[i,0:100])
    
    
P = 0.1
duan = int(len(data)*0.8)

dataX = data[:duan,0:100]
dataY = data[:duan,100:104]
#dataY[:,0] = dataY[:,0]/90

testX = data[duan:,0:100]
testY = data[duan:,100:104]


model = load_model('phmod.h5')
#model = load_model('m3.h5')
model.summary()

predictY = model.predict(testX)

plt.figure(1)
plt.plot(testY[:,1]*100, predictY[:,1],'.')
plt.title('q')
#plt.plot(predictY[:,1])
#plt.plot(testY[:,1]-predictY[:,1])

plt.figure(0)
plt.plot(testY[:,0], predictY[:,0],'.')
plt.title('incl')
#plt.plot(predictY[:,0])
#plt.plot(testY[:,0]-predictY[:,0])


plt.figure(2)
plt.plot(testY[:,2], predictY[:,2],'.')
plt.title('r')
#plt.plot(predictY[:,2])
#plt.plot(testY[:,2]-predictY[:,2])