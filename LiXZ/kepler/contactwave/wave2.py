# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 20:56:21 2021

@author: dingxu
"""
import phoebe
import numpy as np
import matplotlib.pyplot as plt
import imageio


plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#matplotlib画图中中文显示会有问题，需要这两行设置默认字体


logger = phoebe.logger()

b = phoebe.default_binary(contact_binary=True)

#times  = np.linspace(0,1,150)

#b.add_dataset('lc', times=phoebe.linspace(0,1,150), passband= 'LSST:i')#compute_phases
b.add_dataset('rv', compute_phases=phoebe.linspace(0,1,150))
b.add_dataset('lc', times=phoebe.linspace(0,1,150), passband= 'Johnson:B')

b['period@binary'] = 1

#b.set_value('l3_mode', 'fraction')
#b.set_value('l3_frac', 0.9)

b['incl@binary'] =  70
b['q@binary'] =  32.8224*0.01
b['teff@primary'] =  5500  #6208 

b['teff@secondary'] = 5500*100*0.01 #6087

#b['fillout_factor@contact_envelope@envelope@component'] = 0.5

b['sma@binary'] = 1#0.05 2.32
#print(b['sma@binary'])

b['requiv@primary'] = 48.2*0.01    #0.61845703

#b.add_dataset('mesh', times=[0.25], dataset='mesh01')

b.run_compute(irrad_method='none')
print(b['requiv@secondary'])

gif_images = []
plt.figure(0)
fluxmodel = b['value@fluxes@lc01@model']
resultflux = -2.5*np.log10(fluxmodel)
#resultflux1 = resultflux - np.mean(resultflux)
resultflux1 = np.copy(resultflux)

#plt.plot(b['value@times@lc01@model'], resultflux, '.',label="80°")
timesx = b['value@times@lc01@model']
plt.scatter(timesx, resultflux1, s=20, c='b', alpha=0.4)

plt.xlabel('phase',fontsize=14)
plt.ylabel('mag',fontsize=14)

ar7 = np.vstack((timesx,resultflux1))

np.savetxt('mag2.txt', ar7)

flux2 = np.vstack((timesx,fluxmodel))
np.savetxt('flux2.txt', flux2)

for i in range(10):
    print('it is okB2')
    
rvs1 = b.get_value('rvs', component='primary', context='model', dataset='rv01')
rvs2 = b.get_value('rvs', component='secondary', context='model', dataset='rv01')

plt.figure(1)
plt.scatter(timesx, rvs1, s=20, c='b', alpha=0.4)
plt.scatter(timesx, rvs2, s=20, c='b', alpha=0.4)

plt.xlabel('phase',fontsize=14)
plt.ylabel('V(km/s)',fontsize=14)

