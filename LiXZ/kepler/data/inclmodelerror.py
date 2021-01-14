# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 13:06:05 2021

@author: dingxu
"""
from random import shuffle
from tensorflow.keras.models import load_model
#from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing    
model = load_model('incl.hdf5')
#model = load_model('all.hdf5')
#model = load_model('q.hdf5')
#model = load_model('weights-improvement-05354-0.0293.hdf5')
#model = load_model('weights-improvement-03594-0.0379.hdf5') #phoebemodel.h5
#model = load_model('weights-improvement-02175-0.0418.hdf5') #phoebemodel.h5
#model = load_model('weights-improvement-14563-0.0075.hdf5')
model.summary()

data = np.loadtxt('savedatasample3.txt')
'''
dfdata = pd.DataFrame(data)
dfdata = dfdata.sample(n=10000)
npdfdata = np.array(dfdata)
'''
#data = data[data[:,100]>50]

data[:,100] = data[:,100]
data[:,101] = data[:,101]*100
data[:,102] = data[:,102]*100
data[:,103] = data[:,103]*100

for i in range(len(data)):
    data[i,0:100] = -2.5*np.log10(data[i,0:100]) 
    data[i,0:100] = (data[i,0:100])-np.mean(data[i,0:100])
    
    
shuffle(data)

P = 0.9
duan = int(len(data)*P)

dataX = data[:duan,0:100]
dataY = data[:duan,100]
#dataY[:,0] = dataY[:,0]/90

testX = data[duan:,0:100]
testY = data[duan:,100]


predictY = model.predict(testX)
predictdatay = model.predict(dataX)



def displayimg(testY, predictY):
    plt.figure(4)
    plt.clf()
    plt.subplot(221)
    plt.plot(testY[:,0], predictY[:,0],'.')
    plt.title('incl')
    n = np.vstack((testY[:,0], predictY[:,0]))
    np.savetxt('incl.txt', n)

    plt.subplot(222)
    plt.plot(testY[:,1], predictY[:,1],'.')
    plt.title('q')
    n = np.vstack((testY[:,1], predictY[:,1]))
    np.savetxt('q.txt', n)

    plt.subplot(223)
    plt.plot(testY[:,2], predictY[:,2],'.')
    plt.title('r')
    n = np.vstack((testY[:,2], predictY[:,2]))
    np.savetxt('r.txt', n)


    plt.subplot(224)
    plt.plot(testY[:,3], predictY[:,3],'.')
    plt.title('divT')
    n = np.vstack((testY[:,3], predictY[:,3]))
    np.savetxt('divT.txt', n)

    plt.pause(0.1)
    plt.savefig('tu.jpg')

def displayinclimg(testY, predictY):
    plt.plot(testY, predictY[:,0],'.')
    plt.title('incl')
    n = np.vstack((testY, predictY[:,0]))
    np.savetxt('onlyincl.txt', n)
    
    
displayinclimg(testY, predictY)