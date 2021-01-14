# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 18:33:51 2020

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

times  = np.linspace(0,1,150)

#b.add_dataset('lc', times=phoebe.linspace(0,1,100), passband= 'Kepler:mean')#compute_phases
b.add_dataset('lc', times=phoebe.linspace(0,1,150))

b['period@binary'] = 1

b['incl@binary'] =  70
b['q@binary'] =  267*0.01
b['teff@primary'] =  6000  #6208 

b['teff@secondary'] = 6000*100*0.01 #6087

#b['fillout_factor@contact_envelope@envelope@component'] = 0.5

b['sma@binary'] = 1#0.05 2.32
#print(b['sma@binary'])

b['requiv@primary'] = 35*0.01    #0.61845703

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

plt.xlabel('phrase',fontsize=14)
plt.ylabel('mag',fontsize=14)

gif_images.append(imageio.imread('img1.jpg'))
#plt.title('6000k')

#ax = plt.gca()
#ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
#ax.invert_yaxis() #y轴反向

plt.savefig('img1.jpg')
gif_images.append(imageio.imread('img1.jpg'))

b['incl@binary'] =  70
b['q@binary'] =  267*0.01
b['teff@primary'] =  6000  #6208 
b['teff@secondary'] = 6000*100*0.01 #6087
b['requiv@primary'] = 35*0.01
b.run_compute(irrad_method='none')
print(b['requiv@secondary'])
#plt.fig .ure(1)

fluxmodel = b['value@fluxes@lc01@model']
resultflux = -2.5*np.log10(fluxmodel)
resultflux2 = resultflux - np.mean(resultflux)

#plt.plot(b['value@times@lc01@model'], resultflux, '.',label="70°")
plt.scatter(timesx, resultflux2,s=20, c='r', alpha=0.4)

plt.legend(('主星r = 0.43', '主星r = 0.5'), loc='upper right')
#plt.title('5000k')

print(b['requiv@secondary']) 

ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向

plt.savefig('img2.jpg')
gif_images.append(imageio.imread('img2.jpg'))
#plt.title('5000k')

imageio.mimsave("test.gif",gif_images,fps=2)
