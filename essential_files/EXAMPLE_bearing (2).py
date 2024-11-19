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



#Having looked at the spectrum of one of the signals (see first figure), I
#have decided to use a low-pass filter that is flat up to 8MHz and then
#rolls off to zero by 10MHz (i.e. nothing over 10MHz gets past it).
lo_pass_filter_start_roll_off_freq = 8e6
lo_pass_filter_end_roll_off_freq = 10e6

#This is latest time at which echo from near surface of component arrives
#in any signal (obtained by looking at all the signals in the second
#figure)
latest_time_of_first_arrival = 40e-6

#Following obtained by looking at time difference graph in third figure
max_thickness_position_range = np.array([160, 190]) * 1e-3


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

#define a filter
filter_function = nde.fn_hanning_lo_pass(freq.shape[0], lo_pass_filter_start_roll_off_freq / max_freq, lo_pass_filter_end_roll_off_freq / max_freq)

#filter all the signals
filtered_spectra = spectra * filter_function[None, :]

#inverse FFT
filtered_signals = np.fft.ifft(filtered_spectra, fft_pts)

#Plot example raw signal, its frequency spectrum, the filter, and the filtered signal
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3) #create a new figure (fig) containing some axes (ax)
ax1.plot(time * 1e6, voltage[0])
ax1.set_xlabel('Time (us)')
ax1.set_ylabel('Voltage (V)')
ax2.plot(freq / 1e6, np.abs(spectra[0]) / np.max(np.abs(spectra[0])))
ax2.plot(freq / 1e6, np.abs(filter_function),':')
ax2.set_xlabel('Frequency (MHz)')
ax2.set_ylabel('Voltage (V)')
ax3.plot(time * 1e6, np.real(filtered_signals[0]))
ax3.plot(time * 1e6, np.abs(filtered_signals[0]))
ax3.set_xlabel('Time (us)')
ax3.set_ylabel('Voltage (V)')



fig, ax = plt.subplots() #create a new figure (fig) containing some axes (ax)
#Have a look at all the filtered signals side by side
plt.imshow(20 * np.log10(np.abs(filtered_signals) / np.max(np.abs(filtered_signals))),#note conversion to dB values to show weak signals more clearly
               extent = (np.min(time) * 1e6, np.max(time) * 1e6, np.min(pos) * 1e3, np.max(pos) * 1e3), 
               aspect = 'auto', 
               origin = 'lower',
               vmin = -60)
plt.xlabel('Time (us)')
plt.ylabel('Position (mm)')
plt.colorbar()

#We can see that the first arrival signals from the near side of the
#component are all before 40us and (although a bit faint) that all the
#echoes from the far side of the component are later than 40us.
#Therefore if we find time of peaks in each  signal before and after
#40us we can work out the propagation delay through the thickness of the
#component.

#First work out which index corresponds to 40us
latest_index_of_first_arrival = np.argmin(np.abs(time - latest_time_of_first_arrival)) #trick for getting index of nearest point to something

#Find index of peak before 40us in each signal
index1 = np.argmax(abs(filtered_signals[:, 0:latest_index_of_first_arrival]), axis = 1)

#Find index of peak after 40us in each signal
index2 = np.argmax(abs(filtered_signals[:, latest_index_of_first_arrival:]), axis = 1)  
#Above line gives index2 into submatrix of filtered signals so we need to
#add on where the submatrix starts onto index2 so that it is correct for
#original matrix
index2 = index2 + latest_index_of_first_arrival

#Now we can convert to times
time1 = time[index1]
time2 = time[index2]
time_difference = time2 - time1

#Plot calculated echo arrival times and difference between them
fig, ax = plt.subplots() #create a new figure (fig) containing some axes (ax)
plt.plot(pos * 1e3, time1 * 1e6)
plt.plot(pos * 1e3, time2 * 1e6)
plt.plot(pos * 1e3, time_difference * 1e6)
plt.xlabel('Position (mm)')
plt.ylabel('Time (us)')
plt.legend(('First echo', 'Second echo', 'Time difference'))
plt.title('Echo times')

#There are some obvious 'glitches' in the result and there is no universal
#elegant way of getting rid of these. Could look at amplitude of peaks in
#individual signals (probably the glitches are caused by very low amplitude
#second echoes at certain points that are below the noise level, so the 
#position of the maximum value in those signals is random. Alternatively, 
#can look for sudden jumps between adjacent time_difference values and 
#discard those points.

#Ignoring the glitches, it is obvious that the maximum thickness (i.e.
#largest time difference) is somewhere between 160 and 190mm so we can use
#the maximum time difference in this range and the known max thickness of
#the component to work out the velocity.

#Again we can convert these into indices (this time in terms of position)
j1 = np.argmin(abs(pos - max_thickness_position_range[0]))
j2 = np.argmin(abs(pos - max_thickness_position_range[1]))

#Find the max time difference between those two positions
max_time_difference = np.max(time_difference[j1:j2])

#Work out velocity
velocity = 2 * max_thickness / max_time_difference #Factor of two because sound has to get there and back

#Convert time_difference to thickness
thickness_profile  = time_difference * velocity / 2

#Final thickness profile
fig, ax = plt.subplots()
plt.plot(pos * 1e3, thickness_profile * 1e3)
plt.xlabel('Position (mm)')
plt.ylabel('Thickness (mm)')
plt.title('Thickness profile')