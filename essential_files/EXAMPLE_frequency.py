# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 09:46:32 2023

@author: mepdw
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import NDE_functions as nde


#close all open plot windows
plt.close('all')

print('Frequency domain') #display the title

#START OF INPUTS

#pulse parameters
number_cycles = 5 #keep
centre_frequency = 1e6 #keep

#wave propagation
nondispersive_propagation = True #keep
velocity_at_centre_frequency = 3e3 #keep


#distance
distance_step = velocity_at_centre_frequency / centre_frequency / 4#4 pts per wavelength
max_distance = 150e-3 #keep

#time (NB swapped over time and distance inputs so max_distance can be used
#to calc max_time)
t_step = 1 / centre_frequency / 10 #10 pts per cycle at centre freq
max_t = max_distance / velocity_at_centre_frequency + number_cycles / centre_frequency #propagation time + duration of pulse

#END OF INPUTS

"""After this point, everything should be derived from the input values defined 
above - there should be no more hard-coded numbers. This will make code 
easier to debug."""

#START OF PROGRAM 

#create distance vector
distance = np.arange(0, max_distance, step = distance_step) #make a vector of x positions

#create time vector
t = np.arange(0, max_t, step = t_step) #make a vector of times

#create Hanning windowed toneburst
time_at_centre_of_pulse = number_cycles / centre_frequency / 2


window = nde.fn_hanning(t.shape[0], time_at_centre_of_pulse / np.max(t), time_at_centre_of_pulse / np.max(t)) 
sine_wave = np.sin(2 * np.pi * centre_frequency * (t - time_at_centre_of_pulse))

input_signal = window * sine_wave

#At this point the input time-domain signal should have been created

#calculate the frequency spectrum of the input pulse
fft_pts = int(np.ceil(t.shape[0] / 2) * 2) #force even number of points
spectrum = np.fft.fft(input_signal, fft_pts)
spectrum = spectrum[0:int(fft_pts / 2)]

# build frequency axis
f_step = 1 / (fft_pts * t_step)
f = np.arange(spectrum.shape[0]) * f_step

fig, (ax1, ax2) = plt.subplots(nrows = 2) #create a new figure (fig) containing some axes (ax)
ax1.plot(t * 1e6, input_signal)
ax1.set_xlabel('Time (us)')
ax2.plot(f / 1e6, np.abs(spectrum))
ax2.set_xlabel('Frequency (MHz)')

#At this point the frequency spectrum of the input time signal should have been created

if nondispersive_propagation: 
    velocity = velocity_at_centre_frequency #keep
else:
    velocity = np.sqrt(f / centre_frequency) * velocity_at_centre_frequency
    velocity[0] = 1 #To avoid zero / zero warning on line 88

#create a vector of wavenumbers
k = 2 * np.pi * f / velocity 
k[0] = 0 #avoid NaN problem at zero freq

#prepare a matrix to put the results in
P = np.zeros((fft_pts, distance.shape[0]), dtype = np.cdouble)

#loop through the different distances and create the time-domain signal at
#each one and out into the matrix p
for i in range(distance.shape[0]):
    delayed_spectrum = spectrum * np.exp(-1j * k * distance[i]) 
    P[:, i] = np.fft.ifft(delayed_spectrum, fft_pts) * 2

P = P[0:t.shape[0], :]


fig, ax = plt.subplots() #create a new figure (fig) containing some axes (ax)
for i in range(distance.shape[0]):
    ax.cla() #clear the axes
    plt.plot(t * 1e6, np.real(P[:, i]), 'b') #plot the real part of most recent line in blue
    #plt.plot(x, np.imag(P[i,:]), 'r') #plot the imaginary part of most recent line in red
    plt.ylim(-1, 1)
    plt.xlabel('Time (us)') #good practice to add axis label
    plt.title('Distance %.2f mm' % (distance[i] * 1e3))
    fig.canvas.draw() #force the figure to refresh
    fig.canvas.flush_events()
    time.sleep(0.02) #a short pause in seconds before next frame is plotted
