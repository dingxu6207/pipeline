# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 06:37:29 2020

@author: dingxu
"""

import phoebe
import numpy as np
import matplotlib.pyplot as plt



def calculater(ydata, caldata):
    res_ydata  = np.array(ydata) - np.array(caldata)
    ss_res     = np.sum(res_ydata**2)
    ss_tot     = np.sum((ydata - np.mean(ydata))**2)
    r_squared  = 1 - (ss_res / ss_tot)
    return r_squared



logger = phoebe.logger()

b = phoebe.default_binary(contact_binary=True)


lenlight = 150
times  = np.linspace(0,1,lenlight)

#b.add_dataset('lc', times=phoebe.linspace(0,1,100), passband= 'Kepler:mean')#compute_phases
b.add_dataset('lc', times = times)

b['period@binary'] = 1

b['incl@binary'] =  65.26318  #58.528934
b['q@binary'] =     37.003555*0.01
b['teff@primary'] =  6500  #6208 
b['teff@secondary'] = 6500*99.01571*0.01#6500*100.08882*0.01 #6087


#b['fillout_factor@contact_envelope@envelope@component'] = 0.5

b['sma@binary'] = 1#0.05 2.32
#print(b['sma@binary'])

b['requiv@primary'] = 48.471592*0.01    #0.61845703

b.add_dataset('mesh', times=[0.25], dataset='mesh01')

b.run_compute(irrad_method='none')

plt.figure(0)
afig, mplfig = b.plot(show=True, legend=True)

print(b['fillout_factor@contact_envelope'])



np.savetxt('data0.lc', 
           np.vstack((b['value@times@lc01@model'], b['value@fluxes@lc01@model'])).T)


fluxes_model = b['fluxes@model'].interp_value(times=times)
fluxcha = fluxes_model-b['value@times@lc01@model']

#print(fluxcha)

path = 'E:\\shunbianyuan\\data\\kepler\\KIC_name\\'
file = 'KIC 4937217.txt'
#file = 'V396Mon_Yang2001B.nrm'
#path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\pipeline\\LiXZ\\nihe\\'
#yuandata = np.loadtxt(file)
yuandata = np.loadtxt(path+file)
#datay = 10**(yuandata[:,1]/(-2.512))
datay = yuandata[:,1]
#datay = -2.5*np.log10(yuandata[:,1])
datay = datay-np.mean(datay)

#datay = datay/np.mean(datay)

fluxmodel = b['value@fluxes@lc01@model']
resultflux = -2.5*np.log10(fluxmodel)
resultflux = resultflux - np.mean(resultflux)
plt.figure(1)
plt.plot(yuandata[:,0], datay, '.')
plt.scatter(b['value@times@lc01@model'], resultflux, c='none',marker='o',edgecolors='r', s=40)
#plt.plot(b['value@times@lc01@model'], resultflux, '.')
#plt.plot(b['value@times@lc01@model'], -2.5*np.log10(b['value@fluxes@lc01@model'])+0.64, '.')
plt.xlabel('phase',fontsize=14)
plt.ylabel('mag',fontsize=14)

#R2data = calculater(datay, resultflux)
#print('R2data = ', R2data)

ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
'''



phases = b.to_phase(times)
phases_sorted = sorted(phases)
flux = b['fluxes@model'].interp_value(phases=phases_sorted)
'''
'''
from PyAstronomy.pyasl import foldAt
phases = foldAt(times, 3)
sortIndi = np.argsort(phases)
# ... and, second, rearrange the arrays.
phases = phases[sortIndi]
flux = b['value@fluxes@lc01@model'][sortIndi]

np.savetxt('data1.lc', np.vstack((phases, flux)).T)

plt.figure(1)
plt.plot(phases, flux)
'''

'''
for i in range(4):
    b['sma@binary'] = 3.1+0.1*i
    b.run_compute(irrad_method='none')
    plt.figure(i)
    b.plot(show=True, legend=True)
    print(b['fillout_factor@contact_envelope'])

'''