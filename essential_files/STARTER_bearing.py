# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 17:47:33 2023

@author: mepdw
"""

fname = 'bearing_casing_bscan.json'

import numpy as np
import matplotlib.pyplot as plt
import NDE_functions as nde

#close all open plot windows
plt.close('all')

#load bearing data
(pos, time, voltage) = nde.fn_load_bearing_data(fname)

max_thickness = 55e-3 #From exercise sheet

#END OF INPUTS

"""After this point, everything should be derived from the input values defined 
above - there should be no more hard-coded numbers. This will make code 
easier to debug."""

#START OF PROGRAM 
#Do FFTS of all the signals
dt = time[1] - time[0]
fft_pts = time.shape[0]
df = 1 / (fft_pts * dt)

spectra = np.fft.fft(voltage, fft_pts)
spectra = spectra[0:int(fft_pts / 2), :] #discard upper half of spectra
freq = np.arange(spectra.shape[1]) * df
max_freq = np.max(freq)




#Have a look at all the raw signals side by side
plt.imshow(voltage,#note conversion to dB values to show weak signals more clearly
               extent = (np.min(time) * 1e6, np.max(time) * 1e6, np.min(pos) * 1e3, np.max(pos) * 1e3), 
               aspect = 'auto', 
               origin = 'lower')
plt.xlabel('Time (us)')
plt.ylabel('Position (mm)')
plt.colorbar()

#Example of one raw signal
fig, ax1 = plt.subplots() 
ax1.plot(time * 1e6, voltage[0])
ax1.set_xlabel('Time (us)')
ax1.set_ylabel('Voltage (V)')

