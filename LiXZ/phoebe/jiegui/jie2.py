# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 23:43:48 2020

@author: dingxu
"""

import phoebe
import numpy as np
import matplotlib.pyplot as plt
import emcee
import sys
from matplotlib.pyplot import cm 

b = phoebe.default_binary()

b['period@orbit'] = 0.5
b['sma@orbit'] = 3.5
# b['incl@orbit'] = 83.5
# b['requiv@primary'] = 1.2
# b['requiv@secondary'] = 0.8
b['teff@primary'] = 6500.
# b['teff@secondary'] = 5500.

lc = np.loadtxt('data.lc')

b.add_dataset('lc', times=lc[:,0], fluxes=lc[:,1], sigmas=0.05*np.ones(len(lc)))

#flux = b['fluxes@model'].interp_value(phases=phases_sorted)

phoebe.interactive_checks_off()
phoebe.interactive_constraints_off()
b.set_value_all('irrad_method', 'none')

def lnprob(x, adjpars, priors):
    # Check to see that all values are within the allowed limits:
#     if not np.all([priors[i][0] < x[i] < priors[i][1] for i in range(len(priors))]):
#         return -np.inf

    for i in range(len(adjpars)):
        b[adjpars[i]] = x[i]
    
    # Let's assume that our priors are uniform on the range of the physical parameter combinations.
    # This is already handled in Phoebe, which will throw an error if the system is not physical,
    # therefore it's easy to implement the lnprior as =0 when system checks pass and =-inf if they don't.
    # Here we'll 'package' this in a simple try/except statement:
    
    try:
        b.run_compute(irrad_method='none')

        # sum of squares of the residuals
        fluxes_model = b['fluxes@model'].interp_value(times=lc[:,0])
        lnp = -0.5*np.sum((fluxes_model-b['value@fluxes@dataset'])**2 / b['value@sigmas@dataset']**2) 

    except:
        lnp = -np.inf

    sys.stderr.write("lnp = %e\n" % (lnp))

    return lnp

def run(adjpars, priors, nwalkers, niter):
    ndim = len(adjpars)

    p0 = np.array([[p[0] + (p[1]-p[0])*np.random.rand() for p in priors] for i in range(nwalkers)])

#     pool = MPIPool()
#     if not pool.is_master():
#         pool.wait()
#         sys.exit(0)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[adjpars, priors])

    pos, prob, state = sampler.run_mcmc(p0, niter)
    
    return pos


adjpars = ['incl@orbit', 'requiv@primary', 'requiv@secondary', 'teff@secondary']
priors = [(83.0, 84.0), (1.15, 1.3), (0.75, 0.85), (5400., 5500.)]
nwalkers =  10 #32
niters =  14    #10
state = None

import time


#time1 = time.time()
position = run(adjpars, priors, nwalkers, niters)
#time2 = time.time()


mod = b
times = lc[:,0]
color=cm.rainbow(np.linspace(0,1,nwalkers))

for i,c in zip(range(nwalkers),color):
    
    mod['incl@binary@orbit@component'] = position[-1-i,0]
    mod['requiv@primary@star@component'] = position[-1-i,1]
    mod['requiv@secondary@star@component'] = position[-1-i,2]
    mod['teff@secondary'] = position[-1-i,3]    
    mod.run_compute(model='run{}'.format(i))


for i,c in zip(range(nwalkers),color):
    model = mod['fluxes@run{}'.format(i)].interp_value(times=times)

    plt.figure(1)
    plt.plot(times,model,c=c)
plt.plot(times,lc[:,1],"k.")
plt.xlabel("Times")
plt.ylabel("Flux")
