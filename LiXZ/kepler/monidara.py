# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:30:00 2021

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import phoebe


data = np.loadtxt('alldata35.txt')

xuanze = data[data[:,101] >= 0.95] #0.05,95,51,89

flux = xuanze[16,0:100]
labels = xuanze[16,100:104]

redata = -2.5*np.log10(flux)

datay = redata[0:100]-np.mean(redata[0:100])

plt.figure(0)
plt.plot(datay, '.')
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
plt.xlabel('Phase',fontsize=14)
plt.ylabel('mag',fontsize=14)


model = load_model('all.hdf5')

nparraydata = np.reshape(datay,(1,100))

prenpdata = model.predict(nparraydata)

print(prenpdata)
print(labels)

logger = phoebe.logger()

b = phoebe.default_binary(contact_binary=True)

times  = np.linspace(0,1,100)

#b.add_dataset('lc', times=phoebe.linspace(0,1,100), passband= 'Kepler:mean')#compute_phases
b.add_dataset('lc', times=phoebe.linspace(0,1,100))

b['period@binary'] = 1

b['incl@binary'] =  82.71378#58.528934
b['q@binary'] =  99.04949*0.01
b['teff@primary'] =  6500  #6208 
b['teff@secondary'] = 6500*85.1914*0.01#6500*100.08882*0.01 #6087


#b['fillout_factor@contact_envelope@envelope@component'] = 0.5

b['sma@binary'] = 1#0.05 2.32
#print(b['sma@binary'])

b['requiv@primary'] = 39.060974*0.01    #0.61845703

b.add_dataset('mesh', times=[0.25], dataset='mesh01')

b.run_compute(irrad_method='none')

fluxmodel = b['value@fluxes@lc01@model']
resultflux = -2.5*np.log10(fluxmodel)
resultflux = resultflux - np.mean(resultflux)

datax = np.arange(0,100,1)/100
plt.figure(1)
plt.plot(datax, datay)
plt.scatter(b['value@times@lc01@model'], resultflux, c='none',marker='o',edgecolors='r', s=40)
#plt.plot(b['value@times@lc01@model'], resultflux, '.')
#plt.plot(b['value@times@lc01@model'], -2.5*np.log10(b['value@fluxes@lc01@model'])+0.64, '.')
plt.xlabel('phase',fontsize=14)
plt.ylabel('mag',fontsize=14)

ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向

np.savetxt('samle\\yuandata2.txt', xuanze[16,0:104])
np.savetxt('samle\\mag2.txt', resultflux)
