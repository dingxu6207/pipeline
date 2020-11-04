# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 16:59:14 2020

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from keras.optimizers import adam, rmsprop, adadelta

from random import shuffle

data = np.loadtxt('savedata.txt')
data = data[data[:,100]>0]
data[:,100] = data[:,100]/90

#data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
#data[:,0:100] = data_scaler.fit_transform(data[:,0:100])
#scaler = StandardScaler()
#data[:,0:100] = scaler.fit_transform(data[:,0:100])
for i in range(len(data)):
    data[i,0:100] = (data[i,0:100]-np.mean(data[i,0:100]))/np.std(data[i,0:100])


shuffle(data)

P = 0.8
duan = int(len(data)*0.8)

dataX = data[:duan,0:100]
dataY = data[:duan,100:103]
#dataY[:,0] = dataY[:,0]/90

testX = data[duan:,0:100]
testY = data[duan:,100:103]
#testY[:,0] = testY[:,0]/90

models = Sequential()
models.add(Dense(100, init='uniform',activation='relu' ,input_dim=100))
models.add(Dense(50, activation='relu'))
models.add(Dense(30, activation='relu'))
models.add(Dense(20, activation='relu'))
models.add(Dense(3))
#models.add(Dense(3,activation='tanh'))

adamoptimizer = adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.00001)
models.compile(optimizer='rmsprop', loss='mse',metrics=["accuracy"] )


h = models.fit(dataX, dataY, epochs=100, batch_size=10, shuffle=True, verbose = 1)
predictY = models.predict(testX, batch_size=1)
score = models.evaluate(dataX, dataY, batch_size=10)

models.save('phmod.h5')

print(score)

plt.figure(0)
plt.plot(testY[:,1], predictY[:,1],'.')
#plt.plot(predictY[:,1])
#plt.plot(testY[:,1]-predictY[:,1])

plt.figure(1)
plt.plot(testY[:,0], predictY[:,0],'.')
#plt.plot(predictY[:,0])
#plt.plot(testY[:,0]-predictY[:,0])


plt.figure(2)
plt.plot(testY[:,2], predictY[:,2],'.')
#plt.plot(predictY[:,2])
#plt.plot(testY[:,2]-predictY[:,2])


