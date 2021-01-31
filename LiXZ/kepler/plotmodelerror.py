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
import datetime
import time

   
data = np.loadtxt('savedatasample3.txt')

dfdata = pd.DataFrame(data)
dfdata = dfdata.sample(n=100000)
npdfdata = np.array(dfdata)

data = np.copy(npdfdata)
#data = data[data[:,100]>50]

data[:,100] = data[:,100]
data[:,101] = data[:,101]*100
data[:,102] = data[:,102]*100
data[:,103] = data[:,103]*100

for i in range(len(data)):
    data[i,0:100] = -2.5*np.log10(data[i,0:100]) 
    data[i,0:100] = (data[i,0:100])-np.mean(data[i,0:100])
    
    
shuffle(data)

P = 0.7
duan = int(len(data)*P)

dataX = data[:duan,0:100]
#dataY = data[:duan,100:104]
dataY = data[:duan,100]
#dataY[:,0] = dataY[:,0]/90

testX = data[duan:,0:100]
#testY = data[duan:,100:104]
testY = data[duan:,100]

start=time.clock()
#starttime = datetime.datetime.now()
model = load_model('incl.hdf5')
#model = load_model('all.hdf5')
#model = load_model('q.hdf5')
#model = load_model('weights-improvement-05354-0.0293.hdf5')
#model = load_model('weights-improvement-03594-0.0379.hdf5') #phoebemodel.h5
#model = load_model('weights-improvement-02175-0.0418.hdf5') #phoebemodel.h5
#model = load_model('weights-improvement-14563-0.0075.hdf5')
model.summary()
predictY = model.predict(testX)
#predictdatay = model.predict(dataX)
#endtime = datetime.datetime.now()
#print ((endtime - starttime).seconds)
end=time.clock()
total_time=end-start
print("总耗时:"+str(total_time))


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
    
    
def displayincl(testY, predictY):
    plt.figure(0)
    plt.plot(testY, predictY,'.')
    predictY = predictY[:,0]
    sigma = np.std(testY-predictY)
    sigma = round(sigma, 4)
    plt.plot(testY, (testY-predictY)+10, '.', c='blue')
    plt.plot(testY, predictY, '.', color='darkorange')
    plt.plot(testY, testY, '-', c='black')
    plt.axhline(y=10, color='r', linestyle='-')
    plt.axvline(x=45, color='r', linestyle='--')
    plt.text(54, 12, 'y=10  '+'σ='+str(sigma), fontsize=14, color = "b", style = "italic")
    plt.text(45, 4, 'x=45', fontsize=14, color = "r", style = "italic")
    plt.xlabel('incl',fontsize=14)
    plt.ylabel('predict-incl',fontsize=14)
    
displayincl(testY, predictY)