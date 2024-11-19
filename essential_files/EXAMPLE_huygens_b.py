# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 13:47:25 2023

@author: mepdw
"""
import numpy as np
import matplotlib.pyplot as plt

print('Beam profile of a focused array transducer') #display the title

#close all plot windows
plt.close('all')

#Wave parameters
velocity = 6e3
frequency = 5e6

#Transducer details
no_elements = 32
element_pitch = 0.6e-3
element_width = 0.5e-3

#Output grid details
grid_size_x = 50e-3
grid_size_y = 50e-3
grid_pixel_size = 0.1e-3

#Details for displaying output
db_range_for_output = 40

#Focal points to demonstrate
focal_positions = np.array([[-10, 20],
    [0, 30],
    [10, 40]]) * 1e-3

#END OF INPUTS

"""After this point, everything should be derived from the input values defined 
above - there should be no more hard-coded numbers. This will make code 
easier to debug."""

#START OF PROGRAM 

#Set up output grid based on values of grid_size_x, grid_size_y and 
#grid_pixel_size. Start by creating vectors of the necessary x and y values
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

#Set up sources for transducer based on values of transducer_width,
#min_sources_per_wavelength and wavelength (calculated from velocity and
#frequency)
wavelength = velocity / frequency

element_pos_x = np.arange(no_elements) * element_pitch #vector of element x-positions with the right spacing between them but not in right overall place
element_pos_x = element_pos_x - np.mean(element_pos_x) #subtract mean to get them in the right overall position, centred about x = 0
k = 2 * np.pi / wavelength #wavenumber

focal_point_count = 1
for focal_point in focal_positions: #loop over all focal points

    #INSERT LINES FOR THE MAIN CALCULATION HERE
    #Initialise output variable p for this focal point
    p = np.zeros((len(y), len(x))) 
    
    fx = focal_point[0] #position of current focal point
    fy = focal_point[1]
    ref_dist_to_focal_point = np.sqrt(fx ** 2 + fy ** 2) #distance from reference point on array (the centre) to current focal point

    #Main loop
    for  ex in element_pos_x: #loop over the elements in array
        dist_to_focal_point = np.sqrt((ex - fx) ** 2 + fy ** 2) #distance from element to current focal point
        extra_dist = dist_to_focal_point - ref_dist_to_focal_point #extra distance waves must travel from element to current focal point compared to reference distance to focal point
        phase_delay = np.exp(1j * k * extra_dist) #associated phase delay for extra distance
        r = np.sqrt((X - ex) ** 2 + Y ** 2) #distance from current element to every point in output grid
        theta = np.arctan2(X - ex, Y) #angle between element normal and every point in output grid
        #el_directivity = element_width * fn_sinc(0.5 * k * element_width * sin(theta) / pi) #element directivity function. Note that Matlab sinc(x) function evaluates sin(pi*x) / (pi*x) not sin(x)/x, hence need to divide argument by pi
        el_directivity = 1
        q = np.exp(1j * k * r) * el_directivity * np.conj(phase_delay) / np.sqrt(r) #field from current element
        p = p + q #add on to the total field
    #FINISH OF MAIN CALCULATION    
    
    #Plot field in subfigure
    plt.subplot(1, focal_positions.shape[0], focal_point_count)
    focal_point_count += 1
    #As in Ex 4a, rather than normalising by maximum value anywhere in
    #field which can lead to erratic results, here the field is normalised 
    #by the value at the designed focal spot
    nx = np.argmin(np.abs(fx - x))
    ny = np.argmin(np.abs(fy - y))
    amp_at_focal_spot = np.abs(p[ny, nx]) #find amplitude at focal spot
    db_val = 20 * np.log10(np.abs(p) /  amp_at_focal_spot) #pressure field in dB relative to value at focal spot
    plt.imshow(db_val,
               extent = (np.min(x) * 1e3, np.max(x) * 1e3, np.max(y) * 1e3, np.min(y) * 1e3))
    plt.plot(fx *1e3, fy *1e3, 'r+') #red cross at the focal spot
    plt.clim([-db_range_for_output, 0]) #40 dB scale
    plt.colorbar() #add a colour bar with title
    plt.xlabel('x (mm)') #label other axes
    plt.ylabel('y (mm)')
    plt.title('Focus (%.1f, %.1f) mm' % (fx * 1e3, fy * 1e3))
