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

#ADD CODE THAT IS NOT SPECIFIC TO A FOCAL POINT HERE 
#For example, start by creating vectors of the necessary x and y values
x = np.array([0]) #ENTER SUITABLE EQUATION
y = np.array([0]) #ENTER SUITABLE EQUATION


#LOOP OVER THE THREE FOCAL POINTS, DO CALCULATION FOR EACH ONE AND DISPLAY
#RESULTS IN A SUBPLOT

focal_point_count = 1
for focal_point in focal_positions: #loop over all focal points
    #Position of current focal point
    fx = focal_point[0] #position of current focal point
    fy = focal_point[1]
    
    #Initialise output variable p for this focal point
    p = np.zeros((len(y), len(x))) 
    
    #ADD CODE TO DO THE HUYGENS FIELD CALCULATION OF p FOR THE CURRENT
    #FOCAL POINT AT (fx, fy)
    
    #Plot field in subfigure
    plt.subplot(1, focal_positions.shape[0], focal_point_count)
    focal_point_count += 1
    
    #As in Ex 4a, rather than normalising by maximum value anywhere in
    #field which can lead to erratic results, here the field is normalised 
    #by the value at the designed focal spot
    nx = np.argmin(np.abs(fx - x))
    ny = np.argmin(np.abs(fy - y))
    #amp_at_focal_spot = np.abs(p[ny, nx]) #find amplitude at focal spot - commented out at this point as the line won't work unless p is a numpy array
    
    #ADD REST OF PLOTTING CODE
    
