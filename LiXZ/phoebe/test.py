# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 02:19:25 2020

@author: dingxu
"""
import numpy as np
import emcee
import matplotlib.pyplot as pl
import phoebe
import matplotlib.mlab as mlab
from scipy.stats import norm
from matplotlib.pyplot import cm 

b = phoebe.default_binary()

times = np.linspace(0,3,201)
b.add_dataset('lc', times=times, dataset='lc01')

b.get_parameter(context='compute', qualifier='irrad_method').set_value("none")

b.get_parameter(context='compute', component='primary', qualifier='ntriangles').set_value(300)

b.get_parameter(context='compute', component='secondary', qualifier='ntriangles').set_value(300)

b['incl@binary@orbit@component'] = 86.5
b['requiv@primary@star@component'] = 1.2
b['requiv@secondary@star@component'] = 0.8
b['q@binary@orbit@component'] = 0.7
b['pblum@primary@dataset'] = 2.9*np.pi

b.run_compute()

fluxes = b['fluxes@latest@model'].get_value()
sigmas = np.full((len(times)), 0.01)
mean = 0.0
noise = np.random.normal(mean, sigmas, fluxes.shape)
noisy = fluxes + noise

pl.plot(times, noisy, "b.-")
pl.xlabel("Times")
pl.ylabel("Flux")
pl.show()


nwalkers = 10
niter = 2
init_dist = [(86.4,87.3),(1.15,1.25),(0.725,0.825),(0.675,0.725)]

priors = [(80.,90),(1.1,1.3),(0.7,0.9),(0.6,0.9)]

mod = phoebe.default_binary()
mod.add_dataset('lc', times = times, fluxes=fluxes, sigmas=sigmas,compute_phases=np.linspace(0,1.,51), passband='Kepler:mean')
mod.get_parameter(context='compute', qualifier='irrad_method').set_value("none")
mod.get_parameter(context='compute', component='primary', qualifier='ntriangles').set_value(300)
mod.get_parameter(context='compute', component='secondary', qualifier='ntriangles').set_value(300)

mod.set_value('pblum_mode',value='dataset-scaled')

def rpars(init_dist):
    return [np.random.rand() * (i[1]-i[0]) + i[0] for i in init_dist]

def lnprior(priors, values):
    
    lp = 0.
    for value, prior in zip(values, priors):
        if value >= prior[0] and value <= prior[1]:
            lp+=0
        else:
            lp+=-np.inf 
    return lp

def lnprob(z):

    mod['incl@binary@orbit@component'] = z[0]
    mod['requiv@primary@star@component'] = z[1]
    mod['requiv@secondary@star@component'] = z[2]
    mod['q@binary@orbit@component'] = z[3]
    
    lnp = lnprior(priors,z)
    if not np.isfinite(lnp):
            return -np.inf
    
    try: 
        mod.run_compute()

        # use chi^2 to compare the model to the data:
        chi2 = 0.
        for dataset in mod.get_model().datasets:
            chi2+=np.sum(mod.compute_residuals(dataset=dataset, as_quantity=False)**2/sigmas**2)
        # calculate lnprob
        lnprob = -0.5*chi2 + lnp
        return lnprob
    except:
        return -np.inf
 
def run(init_dist, nwalkers, niter):
    # Specify the number of dimensions for mcmc
    ndim = len(init_dist)

    # Generate initial guesses for all parameters for all chains
    p0 = np.array([rpars(init_dist) for i in range(nwalkers)])

    # Generate the emcee sampler. Here the inputs provided include the lnprob function. With this setup, the value z
    # in the lnprob function, is the output from the sampler.
    sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob)
    
    #pos is the position of the walkers in parameter space
    #prob is the probability of the given "pos" positions
    #state is the state of the random number generator
    pos, prob, state = sampler.run_mcmc(p0, niter)

    for i in range(ndim):
        pl.figure()
        y = sampler.flatchain[:,i]
        n, bins, patches = pl.hist(y, 200, normed=1, color="b", alpha=0.45)#, histtype="step")
        pl.title("Dimension {0:d}".format(i))
        
        mu = np.average(y)
        sigma = np.std(y)
        
        print ("mu = ", mu)
        print ("sigma = ",sigmas)

        #bf = mlab.normpdf(bins, mu, sigma)
        bf = norm.pdf(bins, mu, sigma)
        l = pl.plot(bins, bf, 'k--', linewidth=2.0)
    pl.show()
    return pos

pos = run(init_dist, nwalkers, niter)
    
color=cm.rainbow(np.linspace(0,1,nwalkers))
for i,c in zip(range(nwalkers),color):
  
    #pl.figure()
    
    # Set all the parameter values
    mod['incl@binary@orbit@component'] = pos[-1-i,0]
    mod['requiv@primary@star@component'] = pos[-1-i,1]
    mod['requiv@secondary@star@component'] = pos[-1-i,2]
    mod['q@binary@orbit@component'] = pos[-1-i,3]
        
    mod.run_compute(model='run{}'.format(i))


for i,c in zip(range(nwalkers),color):
    model = mod['fluxes@run{}'.format(i)].interp_value(times=times)

    pl.plot(times,model,c=c)

pl.xlabel("Times")
pl.ylabel("Flux")
pl.plot(times,noisy,"k.")
pl.show()

phases = mod.to_phase(times)
phases_sorted = sorted(phases)
for i,c in zip(range(nwalkers),color):
    model = mod['fluxes@run{}'.format(i)].interp_value(phases=phases_sorted)

    pl.plot(phases_sorted,model,c=c)

pl.xlabel("Times")
pl.ylabel("Flux")
pl.plot(phases,noisy,"k.")
pl.show()

