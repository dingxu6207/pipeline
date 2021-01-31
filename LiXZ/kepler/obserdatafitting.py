# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 21:19:01 2021

@author: dingxu
"""

import phoebe
import numpy as np
import matplotlib.pyplot as plt
import random


def calculater(ydata, caldata):
    res_ydata  = np.array(ydata) - np.array(caldata)
    ss_res     = np.sum(res_ydata**2)
    ss_tot     = np.sum((ydata - np.mean(ydata))**2)
    r_squared  = 1 - (ss_res / ss_tot)
    return r_squared



logger = phoebe.logger()

b = phoebe.default_binary(contact_binary=True)


lenlight = 309
times  = np.linspace(0,1,lenlight)


b.add_dataset('lc', times = times)

b['period@binary'] = 1
b['incl@binary'] =  80.41336  #58.528934
b['q@binary'] =    37.9*0.01
b['teff@primary'] =  6500  #6208 
b['teff@secondary'] = 6500*99.8*0.01#6500*100.08882*0.01 #6087
b['sma@binary'] = 1#0.05 2.32


b['requiv@primary'] = 48.4*0.01    #0.61845703

b.add_dataset('mesh', times=[0.25], dataset='mesh01')

b.run_compute(irrad_method='none')

plt.figure(0)
afig, mplfig = b.plot(show=True, legend=True)

print(b['fillout_factor@contact_envelope'])



np.savetxt('data0.lc', np.vstack((b['value@times@lc01@model'], b['value@fluxes@lc01@model'])).T)


fluxes_model = b['fluxes@model'].interp_value(times=times)
fluxcha = fluxes_model-b['value@times@lc01@model']

#path = 'E:\\shunbianyuan\\data\\kepler\\KIC_name\\'
#file = 'CUTau_Qian2005B.nrm'
file = 'V737.txt'#'V396Mon_Yang2001B.nrm'
path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\pipeline\\LiXZ\\nihe\\'
yuandata = np.loadtxt(path+file)
datay = yuandata[:,1]
#datay = -2.5*np.log10(datay)
datay = datay-np.mean(datay)


fluxmodel = b['value@fluxes@lc01@model']
resultflux = -2.5*np.log10(fluxmodel)
resultflux = resultflux - np.mean(resultflux)
plt.figure(1)
plt.plot(yuandata[:,0], datay, '.')
#plt.scatter(b['value@times@lc01@model'], resultflux, c='none',marker='o',edgecolors='r', s=40)
plt.plot(b['value@times@lc01@model'], resultflux, '.')
plt.xlabel('phase',fontsize=14)
plt.ylabel('mag',fontsize=14)
resdata = datay-resultflux
resdata = resdata[resdata<0.05]
resdata = resdata[resdata>-0.05]
xres = np.random.random(len(resdata))
plt.plot(xres,resdata+0.7,'.')
plt.axhline(y=0.7, color='r', linestyle='-')
plt.text(0.018, 0.64, 'y=0.7  '+'residual', fontsize=14, color = "b", style = "italic")


print(np.std(datay-resultflux))
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
