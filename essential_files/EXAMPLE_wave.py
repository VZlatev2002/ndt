# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 10:03:26 2023

@author: mepdw
"""

"""
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
frequency = 1e6 #This value of frequency = 1MHz taken directly from exercise sheet
velocity = 3e3 #This value of velocity = 3,000 m/s taken directly from exercise sheet

#time parameters
t_step = 0.05e-6 #This value of time step = 0.05 us taken directly from exercise sheet
t_points = 100 #This value of animation frames = 100 taken directly from exercise sheet

#distance parameters
number_of_wavelengths = 8 #This value chosen using judgement to show a reasonable number of wavelengths in each frame of the animation
step_in_wavelengths = 1/10 #This value of 1/10 of wavelength step between points in space taken directly from exercise sheet

sign_of_k = 1 #I added these to allow signs of w and k to be changed easily
sign_of_w = -1

#select which parts of exercise to do
do_part_2 = False
do_parts_5_and_6 = False
do_parts_7_and_8 = False
do_part_9 = True

#END OF INPUTS

"""After this point, everything should be derived from the input values defined 
above - there should be no more hard-coded numbers. This will make code 
easier to debug."""

#START OF PROGRAM 

#calculate w (angular frequency), wavelength (wavelength) and k (wavenumber) from inputs
w = 2 * np.pi * frequency #Definition of angular frequency 
wavelength = velocity / frequency #Definition of wavelength in terms of velocity and frequency
k = 2 * np.pi / wavelength #Definition of wavenumber in terms of wavelength

#calculate x (distance) parameters from inputs and value for wavelength
x_end = number_of_wavelengths * wavelength #Physical distance is distance in wavelengths multiplied by wavelength
x_step = step_in_wavelengths * wavelength #Physical distance step is step in wavelengths multiplied by wavelength


x = np.arange(0,x_end, step = x_step) #make a vector of x positions
t = np.arange(0,t_step * t_points, step = t_step) #make a vector of times

if do_part_2:
    #Part 2 of exercise - direct plotting of cos(kx+wt) in a loop
    fig, ax = plt.subplots() #create a new figure (fig) containing some axes (ax)
    
    #the animation loop follows
    for tt in t:
        p = np.cos(sign_of_k * k * x + sign_of_w * w * tt) #Note that for each iteration through the loop we want the result for all values of x, but only the current instance in time, tt
        
        ax.cla() #clear the axes
        plt.plot(x, p) #plot the most recent line
        plt.ylim(-1, 1)
        plt.xlabel('Distance (mm)') #good practice to add axis label
        plt.title('Time %.2f microseconds' % (tt * 1e6))
        fig.canvas.draw() #force the figure to refresh
        fig.canvas.flush_events()
        time.sleep(0.02) #a short pause in seconds before next frame is plotted
    
if do_parts_5_and_6:
    #Parts 5 and 6 of exercise - direct plotting of real and imaginary parts of exp(i(kx+wt)) in a loop
    fig, ax = plt.subplots() #create a new figure (fig) containing some axes (ax)
    
    #the animation loop follows
    for tt in t:
        p = np.exp(1j * (sign_of_k * k * x + sign_of_w * w * tt)) #Note that for each iteration through the loop we want the result for all values of x, but only the current instance in time, tt
        
        ax.cla() #clear the axes
        plt.plot(x, np.real(p), 'b') #plot the real part of most recent line in blue
        plt.plot(x, np.imag(p), 'g') #plot the imaginary part of most recent line in green
        plt.ylim(-1, 1)
        plt.xlabel('Distance (mm)') #good practice to add axis label
        plt.title('Time %.2f microseconds' % (tt * 1e6))
        fig.canvas.draw() #force the figure to refresh
        fig.canvas.flush_events()
        time.sleep(0.02) #a short pause in seconds before next frame is plotted
    

#parts 7, 8 and 9 pre-calculate of p for all values of x and t
(X, T) = np.meshgrid(x, t) #Use meshgrid command to set up equal-sized matrices X and T. Each row of X is a duplicate of x and each column of T is a duplicate of t
P = np.exp(1j * (sign_of_k * k * X + sign_of_w * w * T)) #This is the calculation of P in a single line. P will be matrix of same size as X and T

if do_parts_7_and_8:
    #Part 7 and 8 of exercise - animation of pre-calculated values
    fig, ax = plt.subplots() #create a new figure (fig) containing some axes (ax)
    for i in range(t.shape[0]):
        ax.cla() #clear the axes
        plt.plot(x, np.real(P[i,:]), 'b') #plot the real part of most recent line in blue
        plt.plot(x, np.imag(P[i,:]), 'g') #plot the imaginary part of most recent line in green
        plt.ylim(-1, 1)
        plt.xlabel('Distance (mm)') #good practice to add axis label
        plt.title('Time %.2f microseconds' % (t[i] * 1e6))
        fig.canvas.draw() #force the figure to refresh
        fig.canvas.flush_events()
        time.sleep(0.02) #a short pause in seconds before next frame is plotted

if do_part_9:
    #surface plot of pre-calculated values
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}) #create a new figure (fig) containing some axes (ax)
    ax.plot_surface(X * 1e3, T * 1e6, np.real(P))
    plt.xlabel('Distance (mm)') #good practice to add axis label
    plt.ylabel('Time (us)') #good practice to add axis label
    
    #2d image plot of pre-calculated values
    fig, ax = plt.subplots() #create a new figure (fig) containing some axes (ax)
    plt.imshow(np.real(P), 
               extent = (np.min(x) * 1e3, np.max(x) * 1e3, np.min(t) * 1e6, np.max(t) * 1e6), 
               aspect = 'auto', 
               origin = 'lower')
    plt.xlabel('Distance (mm)') #good practice to add axis label
    plt.ylabel('Time (us)') #good practice to add axis label
    
