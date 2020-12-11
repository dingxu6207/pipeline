#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 09:53:02 2020

@author: dingxu
"""
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
#from keras.optimizers import adam, rmsprop, adadelta

from random import shuffle

data = np.loadtxt('savedata.txt')
print(len(data))
data = data[data[:,100]>40]
#data[:,100] = data[:,100]/90
data[:,100] = data[:,100]
data[:,101] = data[:,101]*100
data[:,102] = data[:,102]*100
data[:,103] = data[:,103]/10

for i in range(len(data)):
    data[i,0:100] = -2.5*np.log10(data[i,0:100]) 
    data[i,0:100] = (data[i,0:100])-np.mean(data[i,0:100])
    
    
shuffle(data)

P = 0.8
duan = int(len(data)*0.8)

dataX = data[:duan,0:100]
dataY = data[:duan,100:104]
#dataY[:,0] = dataY[:,0]/90

testX = data[duan:,0:100]
testY = data[duan:,100:104]
#testY[:,0] = testY[:,0]/90

models = Sequential()
models.add(Dense(200,activation='relu' ,input_dim=100))
models.add(Dense(160, activation='relu'))
models.add(Dense(100, activation='relu'))
models.add(Dense(40, activation='relu'))
models.add(Dense(4))
#models.add(Dense(3,activation='tanh'))

#adamoptimizer = adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.00001)
models.compile(loss='mse', optimizer='adam', metrics=['accuracy'])


h = models.fit(dataX, dataY, epochs=2000, batch_size=10, verbose = 1)
predictY = models.predict(testX)
score = models.evaluate(dataX, dataY, batch_size=10)

models.save('phmod.h5')
print(score)

plt.figure(1)
plt.plot(testY[:,1], predictY[:,1],'.')
#plt.plot(predictY[:,1])
#plt.plot(testY[:,1]-predictY[:,1])

plt.figure(0)
plt.plot(testY[:,0], predictY[:,0],'.')
#plt.plot(predictY[:,0])
#plt.plot(testY[:,0]-predictY[:,0])


plt.figure(2)
plt.plot(testY[:,2], predictY[:,2],'.')
#plt.plot(predictY[:,2])
#plt.plot(testY[:,2]-predictY[:,2])

plt.figure(3)
plt.plot(testY[:,3], predictY[:,3],'.')
