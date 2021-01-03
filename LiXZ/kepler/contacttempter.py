# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 18:33:51 2020

@author: dingxu
"""

import phoebe
import numpy as np
import matplotlib.pyplot as plt
import imageio

logger = phoebe.logger()

b = phoebe.default_binary(contact_binary=True)

times  = np.linspace(0,1,150)

#b.add_dataset('lc', times=phoebe.linspace(0,1,100), passband= 'Kepler:mean')#compute_phases
b.add_dataset('lc', times=phoebe.linspace(0,1,150))

b['period@binary'] = 1

b['incl@binary'] =  80.08944
b['q@binary'] =  43.035305*0.01
b['teff@primary'] =  6500  #6208 

b['teff@secondary'] = 6500*90*0.01 #6087

#b['fillout_factor@contact_envelope@envelope@component'] = 0.5

b['sma@binary'] = 1#0.05 2.32
#print(b['sma@binary'])

b['requiv@primary'] = 47.23491*0.01    #0.61845703

b.add_dataset('mesh', times=[0.25], dataset='mesh01')

b.run_compute(irrad_method='none')

gif_images = []
plt.figure(0)
afig, mplfig = b.plot(show=True, legend=True)
plt.savefig('img1.jpg')
gif_images.append(imageio.imread('img1.jpg'))
plt.title('6000k')


b['teff@primary'] =  5000  #6208 
b['teff@secondary'] = 5000*90*0.01 #6087

b.run_compute(irrad_method='none')

plt.figure(1)
afig, mplfig = b.plot(show=True, legend=True)
plt.savefig('img2.jpg')
gif_images.append(imageio.imread('img2.jpg'))
plt.title('5000k')

imageio.mimsave("test.gif",gif_images,fps=2)
