# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 19:36:45 2021

@author: dingxu
"""

import phoebe
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.logger()

b = phoebe.default_binary()
b.add_constraint('semidetached', 'secondary')

times  = np.linspace(0,1,100)

#b.add_dataset('lc', times=phoebe.linspace(0,1,100), passband= 'Kepler:mean')#compute_phases
b.add_dataset('lc', times=times)

b['period@binary'] = 1
b['sma@binary'] = 1#0.05 2.32

b['incl@binary'] =  69 #58.528934
b['q@binary'] =     0.28
b['teff@primary'] =  7400  #6208 
b['teff@secondary'] = 5116.57#6500*100.08882*0.01 #6087
#b['requiv@primary'] = 0.1
b['requiv@primary'] = 0.25

#b['gravb_bol@primary@component'] = 0.5
#b['irrad_frac_refl_bol@primary'] = 0.5
#
#b['gravb_bol@secondary@component'] = 0.5
#b['irrad_frac_refl_bol@secondary'] = 0.32

#b.flip_constraint('pot', solve_for='requiv@primary')
#b.flip_constraint('fillout_factor', solve_for='pot')
#b['fillout_factor'] = 18.668695*0.01
#b['fillout_factor@contact_envelope'] = 0.5

#b['fillout_factor@contact_envelope@envelope@component'] = 0.5

'''
b.get_constraint(qualifier='fillout_factor@contact_envelope@envelope@component')
b['fillout_factor@contact_envelope@envelope@component'] = 0.3
b.flip_constraint(qualifier='fillout_factor@contact_envelope@envelope@component', solve_for='requiv@primary')
'''
#b.get_constraint(qualifier='fillout_factor')
#b.flip_constraint(qualifier='requiv@secondary@component', solve_for='requiv@primary@component')
#b.flip_constraint(qualifier='fillout_factor', context='contact_envelope', solve_for='requiv@primary@star@component')
#b['fillout_factor'] = 0.3

b.add_dataset('mesh', times=[0.25], dataset='mesh01')

b.run_compute()

plt.figure(0)
afig, mplfig = b.plot(show=True, legend=True)

print(b['fillout_factor@contact_envelope'])

#print(b.filter(context='constraint').qualifiers)
#print(b.filter(context='component', kind='star', component='primary'))
#print(b.filter(component='binary'))
#print(b.filter(context='component'))
print(b.filter(context='component', kind='envelope'))
#print(b.filter(context='system'))
print(b.compute_pblums())
#print(b.filter('pblum*'))
print(b.filter('r*'))

print(b.filter('l3*'))

np.savetxt('data0.lc', 
           np.vstack((b['value@times@lc01@model'], b['value@fluxes@lc01@model'])).T)


fluxes_model = b['fluxes@model'].interp_value(times=times)
fluxcha = fluxes_model-b['value@times@lc01@model']

#print(fluxcha)

#path = 'E:\\shunbianyuan\\data\\kepler\\KIC_name\\'
#file = 'KIC 4937217.txt'
file = 'cxh.txt'
#path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\pipeline\\LiXZ\\nihe\\'
yuandata = np.loadtxt(file)
#yuandata = np.loadtxt(path+file)
#datay = 10**(yuandata[:,1]/(-2.512))
datay = yuandata[:,1]
#datay = -2.5*np.log10(yuandata[:,1])
datay = datay-np.mean(datay)

#datay = datay/np.mean(datay)

fluxmodel = b['value@fluxes@lc01@model']
resultflux = -2.5*np.log10(fluxmodel)
resultflux = resultflux - np.mean(resultflux)+0.02
plt.figure(1)
plt.plot(yuandata[:,0], datay, '.')
#plt.scatter(b['value@times@lc01@model'], resultflux, c='none',marker='o',edgecolors='r', s=80)
plt.plot(b['value@times@lc01@model'], resultflux, '.')
#plt.plot(b['value@times@lc01@model'], -2.5*np.log10(b['value@fluxes@lc01@model'])+0.64, '.')
plt.xlabel('phase',fontsize=14)
plt.ylabel('mag',fontsize=14)


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