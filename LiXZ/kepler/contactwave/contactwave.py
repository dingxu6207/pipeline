# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 20:16:31 2021

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

#b.add_dataset('lc', times=phoebe.linspace(0,1,100), passband= 'Kepler:mean')#compute_phases
b.add_dataset('lc', times=phoebe.linspace(0,1,150), passband= 'Johnson:I')

b['period@binary'] = 1

b.set_value('l3_mode', 'fraction')
b.set_value('l3_frac', 0.3)

b['incl@binary'] =  57.8
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
resultflux1 = resultflux - np.mean(resultflux)

#plt.plot(b['value@times@lc01@model'], resultflux, '.',label="80°")
timesx = b['value@times@lc01@model']
plt.scatter(timesx, resultflux1, s=20, c='b', alpha=0.4)

plt.xlabel('phase',fontsize=14)
plt.ylabel('mag',fontsize=14)

#gif_images.append(imageio.imread('img1.jpg'))
##plt.title('6000k')
#
##ax = plt.gca()
##ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
##ax.invert_yaxis() #y轴反向
#
#plt.savefig('img1.jpg')
#gif_images.append(imageio.imread('img1.jpg'))


bc = phoebe.default_binary(contact_binary=True)
bc.add_dataset('lc', times=phoebe.linspace(0,1,100), passband= 'Johnson:U')

bc['period@binary'] = 1
bc['incl@binary'] =  57.8
bc['q@binary'] =  32.8224*0.01
bc['teff@primary'] =  5500  #6208 
bc['teff@secondary'] = 5500*100*0.01 #6087
bc['requiv@primary'] =  52*0.01

bc.set_value('l3_mode', 'fraction')
bc.set_value('l3_frac', 0.3)

bc.run_compute(irrad_method='none')
print(bc['requiv@secondary'])
#plt.fig .ure(1)

fluxmodel = bc['value@fluxes@lc01@model']
resultflux = -2.5*np.log10(fluxmodel)
resultflux2 = resultflux - np.mean(resultflux)

#plt.plot(b['value@times@lc01@model'], resultflux, '.',label="70°")
timesx = bc['value@times@lc01@model']
plt.scatter(timesx, resultflux2,s=20, c='r', alpha=0.4)

plt.legend(('Johnson:I', 'Johnson:U'), loc='upper right')
#plt.title('5000k')

print(b['requiv@secondary']) 

ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向

#plt.savefig('img2.jpg')
#gif_images.append(imageio.imread('img2.jpg'))
##plt.title('5000k')
#
#imageio.mimsave("test.gif",gif_images,fps=2)
