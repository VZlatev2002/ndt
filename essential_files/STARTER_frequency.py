# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 09:46:32 2023

@author: mepdw
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import NDE_functions as nde
import sys


#close all open plot windows
plt.close('all')

print('Frequency domain') #display the title

#START OF INPUTS

#pulse parameters
number_cycles = 5 #keep
centre_frequency = 1e6 #keep
time_for_one_cycle = 1 / centre_frequency

#wave propagation
nondispersive_propagation = True #keep
velocity_at_centre_frequency = 3e3 #keep

#distance
distance_step = velocity_at_centre_frequency / centre_frequency / 4#4 pts per wavelength
max_distance = 150e-3 #keep

#time
t_step = time_for_one_cycle / 10  #ENTER SUITABLE NUMERIC VALUE

max_t = max_distance / velocity_at_centre_frequency + number_cycles * time_for_one_cycle #ENTER SUITABLE NUMERIC VALUE


"""After this point, everything should be derived from the input values defined 
above - there should be no more hard-coded numbers. This will make code 
easier to debug."""

#START OF PROGRAM 

#create distance vector
distance = np.arange(0, max_distance, step = distance_step) #ENTER SUITABLE EQUATION

#create time vector
t = np.arange(0, max_t, step = t_step) #ENTER SUITABLE EQUATION

#create Hanning windowed toneburst
time_at_centre_of_pulse = number_cycles / centre_frequency / 2 #ENTER SUITABLE EQUATION

window = 0 #ENTER SUITABLE EQUATION
sine_wave = 0 #ENTER SUITABLE EQUATION

input_signal = window * sine_wave

#At this point the input time-domain signal should have been created
sys.exit() #This stops the program here to allow you to look at input signal. 
#Once it's working up to this point, remove the exit() line to allow it to continue

#calculate the frequency spectrum of the input pulse
fft_pts = 0 #ENTER SUITABLE EQUATION
spectrum = 0 #ENTER SUITABLE EQUATION
spectrum = 0 #ENTER SUITABLE EQUATION

# build frequency axis
f_step = 0 #ENTER SUITABLE EQUATION
f = 0 #ENTER SUITABLE EQUATION

#At this point the frequency spectrum of the input time signal should have been created
sys.exit() #This stops the program here to allow you to look at input signal spectrum. 
#Once it's working up to this point, remove the exit() line to allow it to continue

if nondispersive_propagation: 
    velocity = velocity_at_centre_frequency #keep
else:
    velocity = 0 #ENTER SUITABLE EQUATION

#create a vector of wavenumbers
k = 0 #ENTER SUITABLE EQUATION 

#prepare a matrix to put the results in
P = 0 #ENTER SUITABLE EQUATION

#loop through the different distances and create the time-domain signal at
#each one and out into the matrix p
for i in range(distance.shape[0]):
    delayed_spectrum = 0 #ENTER SUITABLE EQUATION 
    #ADD THE NECESSARY LINES TO FILL P UP WITH VALUES

fig, ax = plt.subplots() #create a new figure (fig) containing some axes (ax)
for i in range(distance.shape[0]):
    ax.cla() #clear the axes
    #ADD THE NECESSARY LINES TO ANIMATE HOW THE TIME SIGNAL CHANGES AS PROPAGATION DISTANCE INCREASES
    fig.canvas.draw() #force the figure to refresh
    fig.canvas.flush_events()
    time.sleep(0.02) #a short pause in seconds before next frame is plotted
