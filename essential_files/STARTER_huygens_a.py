# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 10:34:30 2023

@author: mepdw
"""
import numpy as np
import matplotlib.pyplot as plt

print('Beam profile of a 2D transducer') #display the title

#close all plot windows
plt.close('all')

#INPUTS

#Wave parameters
velocity = 6e3
frequency = 1e6

#Transducer details
transducer_width = 20e-3
min_sources_per_wavelength = 5

#Output grid details
grid_size_x = 100e-3
grid_size_y = 150e-3
grid_pixel_size = 0.1e-3

#Details for displaying output
normalise_by_max_value_below_this_depth = 10e-3
db_range_for_output = 40

#END OF INPUTS

"""After this point, everything should be derived from the input values defined 
above - there should be no more hard-coded numbers. This will make code 
easier to debug."""

#START OF PROGRAM 

#Set up output grid based on values of grid_size_x, grid_size_y and 
#grid_pixel_size. Start by creating vectors of the necessary x and y values
x = np.array([0]) #ENTER SUITABLE EQUATION
y = np.array([0]) #ENTER SUITABLE EQUATION

#Set up source positions to represent transducer based on values of 
#transducer_width and min_sources_per_wavelength
source_x_positions = np.array([0]) #ENTER SUITABLE EQUATION

#Initialise p to hold output of simulation
p = np.zeros((len(y), len(x))) 

#Main loop
for sx in source_x_positions:
    p = p + 0 #ENTER SUITABLE EQUATION


#Plot output
#Calculate row index of start of region of output in which to find max
#value to normalise by
depth_index = np.argmin(np.abs(normalise_by_max_value_below_this_depth - y))
#Find max value in this region of output
value_to_normalise_by = np.max(np.max(np.abs(p[depth_index:, :])))
#Note: you can just normalise by the largest value anywhere in p (i.e.
#max(abs(p(:)))) but this can give erratic results if one of the grid
#locations happens to be very close to a source location because of the
#1/sqrt(r) term - you will get a very high amplitude field at this point
#and everything else will be off the bottom of the colour scale!

db_val = 20 * np.log10(np.abs(p) / value_to_normalise_by) #Pressure field in dB relative to normalisation value
plt.imshow(db_val,
           extent = (np.min(x) * 1e3, np.max(x) * 1e3, np.max(y) * 1e3, np.min(y) * 1e3))
#axis equal axis tight 
#caxis([-db_range_for_output, 0]) #40 dB scale
plt.clim(-40, 0)
plt.colorbar()
plt.xlabel('x (mm)') #label other axes
plt.ylabel('y (mm)')