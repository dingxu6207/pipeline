#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 00:05:40 2020

@author: dingxu
"""


import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
#from keras.optimizers import adam, rmsprop, adadelta

from random import shuffle

data = np.loadtxt('savedatasample2.txt')

data = data[data[:,100]>70]
data = data[data[:,101]<0.4]
data = data[data[:,103]<1.2]
print(len(data))


data[:,100] = data[:,100]
data[:,101] = data[:,101]*100
data[:,102] = data[:,102]*100
data[:,103] = data[:,103]*100

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
#testY[:,0] = testY[:,0]/90aaa

models = Sequential()
models.add(Dense(200,activation='relu' ,input_dim=100))
models.add(Dense(160, activation='relu'))
models.add(Dense(100, activation='relu'))
models.add(Dense(40, activation='relu'))
models.add(Dense(4))
#models.add(Dense(3,activation='tanh'))

#adamoptimizer = adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.00001)
models.compile(loss='mse', optimizer='adam', metrics=['accuracy'])


h = models.fit(dataX, dataY, epochs=7000, batch_size=15, verbose = 1)
predictY = models.predict(testX)
score = models.evaluate(dataX, dataY, batch_size=10)

models.save('phmodsample2.h5')
print(score)

plt.figure(1)
plt.plot(testY[:,1], predictY[:,1],'.')
#plt.plot(testY[:,1], testY[:,1]+0.5,label = 'y=x+0.5')
plt.plot(testY[:,1], testY[:,1],label = 'y=x')
#plt.plot(testY[:,1], testY[:,1]-0.25,label = 'y=x-0.5')
plt.title('q')
n = np.vstack((testY[:,1], testY[:,1]))
np.savetxt('q.txt', n)
#plt.plot(predictY[:,1])
#plt.plot(testY[:,1]-predictY[:,1])

plt.figure(0)
plt.plot(testY[:,0], predictY[:,0],'.')
#plt.plot(testY[:,0], testY[:,0]+0.5, label = 'y=x+0.5')
#plt.plot(testY[:,0], testY[:,0], label = 'y=x')
#plt.plot(testY[:,0], testY[:,0]-1,label = 'y=x-1')
plt.title('incl')
n = np.vstack((testY[:,0], testY[:,0]))
np.savetxt('incl.txt', n)
#plt.plot(predictY[:,0])
#plt.plot(testY[:,0]-predictY[:,0])


plt.figure(2)
plt.plot(testY[:,2], predictY[:,2],'.')
plt.title('r')
n = np.vstack((testY[:,2], testY[:,2]))
np.savetxt('r.txt', n)
#plt.plot(predictY[:,2])
#plt.plot(testY[:,2]-predictY[:,2])

plt.figure(3)
plt.plot(testY[:,3], predictY[:,3],'.')
plt.title('delT')
n = np.vstack((testY[:,3], testY[:,3]))
np.savetxt('delT.txt', n)
