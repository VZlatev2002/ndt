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
#grid_pixel_size
grid_pts_x = round(grid_size_x / grid_pixel_size)
grid_pts_y = round(grid_size_y / grid_pixel_size)
x = np.linspace(-grid_size_x / 2, grid_size_x / 2, grid_pts_x)
y = np.linspace(0, grid_size_y, grid_pts_y)
#NB There is no one correct way of setting up these vectors as in general
#both size limits and pixel_size cannot both be satisfied. Above method forces 
#start and end points to be exactly as specified, but pixel size in x and y
#may not be exactly as specified. An alternative is 
#   x=-grid_size_x/2:grid_pixel_size:grid_size_x/2
#This will give exact pixel size, but end point might not be quite right

(X, Y) = np.meshgrid(x, y) #useful function to produce matrices X and Y from vectors x and y. See Matlab help

#Set up source positions to represent transducer based on values of 
#transducer_width and min_sources_per_wavelength
wavelength = velocity / frequency
transducer_pts = round(transducer_width / wavelength * min_sources_per_wavelength)
source_x_positions = np.linspace(-transducer_width/2, transducer_width/2, transducer_pts)

#Initialise p to hold output of simulation
p = np.zeros((len(y), len(x))) 

#INSERT LINES FOR THE MAIN CALCULATION HERE
dx = source_x_positions[1] - source_x_positions[0] #length of physical transducer represented by each source
k = 2 * np.pi / wavelength #wavenumber

#Main loop
for sx in source_x_positions:
    r = np.sqrt((X - sx) ** 2 + Y ** 2) #distance from current source to every point in output grid
    q = np.exp(1j * k * r) / np.sqrt(r) * dx #field from current source
    p = p + q #add on to the total field
#FINISH OF MAIN CALCUALTION


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