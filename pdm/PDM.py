# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 16:03:28 2020

@author: dingxu
"""

from PyAstronomy.pyTiming import pyPDM
import numpy
import matplotlib.pylab as plt
from PyAstronomy.pyasl import foldAt
import numpy as np

scanner = pyPDM.Scanner(minVal=0.5, maxVal=1.0, dVal=0.05, mode="period")

print("Periods: ", end=' ')
for period in scanner:
    print(period, end=' ')
    
# Create artificial data with frequency = 3,
# period = 1/3
x = numpy.arange(100) / 100.0
y = numpy.sin(x*2.0*numpy.pi*2.0 + 1.7)

# Get a ``scanner'', which defines the frequency interval to be checked.
# Alternatively, also periods could be used instead of frequency.
S = pyPDM.Scanner(minVal=1, maxVal=15.0, dVal=0.05, mode="frequency")



# Carry out PDM analysis. Get frequency array
# (f, note that it is frequency, because the scanner's
# mode is ``frequency'') and associated Theta statistic (t).
# Use 10 phase bins and 3 covers (= phase-shifted set of bins).
P = pyPDM.PyPDM(x, y)
f1, t1 = P.pdmEquiBinCover(10, 3, S)
# For comparison, carry out PDM analysis using 10 bins equidistant
# bins (no covers).
f2, t2 = P.pdmEquiBin(10, S)


phases = foldAt(x, 0.5)

sortIndi = np.argsort(phases)
# ... and, second, rearrange the arrays.
phases = phases[sortIndi]
flux = y[sortIndi]

# Show the result
plt.figure(0)
plt.title("Result of PDM analysis")
plt.xlabel("Frequency")
plt.ylabel("Theta")
plt.plot(f1, t1, 'bp-')
plt.plot(f2, t2, 'gp-')
plt.legend(["pdmEquiBinCover", "pdmEquiBin"])
plt.show()

plt.figure(1)
plt.plot(phases, flux, 'bp')

