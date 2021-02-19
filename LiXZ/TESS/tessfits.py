# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 00:42:01 2021

@author: dingxu
"""

#tess2020351194500-s0033-0000000805897029-0203-s_tp.fits

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

#dvt_file = "https://archive.stsci.edu/missions/tess/tid/s0002/0000/0001/0010/0827/tess2018235142541-s0002-s0002-0000000100100827-00109_dvt.fits"

dvt_file = 'https://archive.stsci.edu/missions/tess/tid/s0033/0000/0008/0589/7029/tess2020351194500-s0033-0000000805897029-0203-s_tp.fits'
fits.info(dvt_file)

fits.getdata(dvt_file, ext=1).columns

with fits.open(dvt_file, mode="readonly") as hdulist:
    
    # Extract stellar parameters from the primary header.  We'll get the effective temperature, surface gravity,
    # and TESS magnitude.
    star_teff = hdulist[0].header['TEFF']
    star_logg = hdulist[0].header['LOGG']
    star_tmag = hdulist[0].header['TESSMAG']
    
    # Extract some of the fit parameters for the first TCE.  These are stored in the FITS header of the first
    # extension.
    #period = hdulist[1].header['TPERIOD']
    #duration = hdulist[1].header['TDUR']
    #epoch = hdulist[1].header['TEPOCH']
    #depth = hdulist[1].header['TDEPTH']
    
    # Extract some of the columns of interest for the first TCE signal.  These are stored in the binary FITS table
    # in the first extension.  We'll extract the timestamps in TBJD, phase, initial fluxes, and corresponding
    # model fluxes.
    times = hdulist[1].data['TIME']
    #phases = hdulist[1].data['PHASE']
    fluxes_init = hdulist[1].data['LC_INIT']
    #model_fluxes_init = hdulist[1].data['MODEL_INIT']
    
    
    
    # First sort the phase and flux arrays by phase so we can draw the connecting lines between points.
sort_indexes = np.argsort(phases)

# Start figure and axis.
fig, ax = plt.subplots(figsize=(12,4))

# Plot the detrended fluxes as black circles.  We will plot them in sorted order.
ax.plot(phases[sort_indexes], fluxes_init[sort_indexes], 'ko',
       markersize=2)

# Plot the model fluxes as a red line.  We will plot them in sorted order so the line connects between points cleanly.
ax.plot(phases[sort_indexes], model_fluxes_init[sort_indexes], '-r')

# Let's label the axes and define a title for the figure.
fig.suptitle('TIC 100100827 - Folded Lightcurve And Transit Model.')
ax.set_ylabel("Flux (relative)")
ax.set_xlabel("Orbital Phase")

# Let's add some text in the top-right containing some of the fit parameters.
plt.text(0.2, 0.012, "Period = {0:10.6f} days".format(period))
plt.text(0.2, 0.010, "Duration = {0:10.6f} hours".format(duration))
plt.text(0.2, 0.008, "Depth = {0:10.6f} ppm".format(depth))
plt.text(0.45, 0.012, "Star Teff = {0:10.6f} K".format(star_teff))
plt.text(0.45, 0.010, "Star log(g) = {0:10.6f}".format(star_logg))

plt.show()


plt.figure(2)
plt.plot(times, fluxes_init)