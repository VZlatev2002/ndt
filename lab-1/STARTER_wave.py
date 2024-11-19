# -*- coding: utf-8 -*-
"""
CONTINUOUS 1D WAVE PROPAGATION

NOTE: This programme is supposed to produce animations of wave propagation. 
If using Python in the Spyder IDE, you need to make sure it outputs graphics to
a separate window rather than just displaying the final frame of the animation 
in the Spyder "Plots" window.

To ensure this is the case, select:

Tools > Preferences > IPython console > Graphics > Graphics backend > Automatic

rather than:

Tools > Preferences > IPython console > Graphics > Graphics backend > Inline
"""

import numpy as np
import matplotlib.pyplot as plt
import time

print('Wave Animation 1') #display the title

#close all plot windows
plt.close('all')

#START OF INPUTS

#wave parameters
frequency = 10000 #ENTER SUITABLE NUMERIC VALUE
velocity = 10 #ENTER SUITABLE NUMERIC VALUE

#time parameters
t_step = 0.1 #ENTER SUITABLE NUMERIC VALUE
t_points = 10000 #ENTER SUITABLE NUMERIC VALUE

#distance parameters
number_of_wavelengths = 10 #ENTER SUITABLE NUMERIC VALUE
step_in_wavelengths = 10 #ENTER SUITABLE NUMERIC VALUE

#END OF INPUTS

"""After this point, everything should be derived from the input values defined 
above - there should be no more hard-coded numbers. This will make code 
easier to debug."""

#START OF PROGRAM 

# Define parameters
frequency = 1e6  # 1 MHz
velocity = 3000  # 3000 m/s
wavelength = velocity / frequency  # Calculate wavelength
angular_frequency = 2 * np.pi * frequency  # Angular frequency
wavenumber = 2 * np.pi / wavelength  # Wavenumber

# Calculate maximum distance and distance step
x_end = 8 * wavelength  # 8 wavelengths long
x_step = wavelength / 10  # Distance between points < 1/10 of a wavelength
# Create vectors of equally spaced values for distance (x) and time (t)
x = np.arange(0, x_end, step=x_step)
t_step = 0.05e-6  # 0.05 µs
t_points = 100  # 100 frames
t = np.arange(0, t_step * t_points, step = t_step)

#create a new figure (fig) containing some axes (ax)
fig, ax = plt.subplots() 

#loop through each time value in t. At each instant in time
#it to create an animation 
for tt in t:
    p_imag = np.imag(np.exp(1j * (wavenumber * x - angular_frequency * tt)))  # Wave equationnction of distance and plot 
    p_real = np.real(np.exp(1j * (wavenumber * x - angular_frequency * tt)))
    
    ax.cla() #clear the axes
    plt.plot(x, p_imag ,  color = 'red') #plot the most recent line
    plt.plot(x, p_real ,  color = 'blue') #plot the most recent line
    
    
    plt.ylim(-1, 1)
    plt.xlabel('Distance (mm)') #good practice to add axis label
    plt.title('Time %.2f microseconds' % (tt * 1e6))
    fig.canvas.draw() #force the figure to refresh
    fig.canvas.flush_events()
    time.sleep(0.02) #a short pause in seconds before next frame is plotted
