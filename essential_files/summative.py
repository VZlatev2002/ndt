
import NDE_functions as nde
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

plt.close('all')

# Array parameters
no_elements = 32
element_pitch = 0.6e-3
element_width = 0.5e-3
centre_frequency = 5e6
time_pts = 1000

# Imaging parameters
grid_size_x = 50e-3
grid_size_y = 50e-3
grid_pixel_size = 0.5e-3


# Wave parameters
velocity = 6e3

# START OF PROGRAM 
# Generate raw data
(fmc_data, time, element_positions) = nde.fn_simulate_data_ex5_v2(no_elements, element_pitch, element_width, centre_frequency, time_pts)

# Plot a typical raw signal (transmit and receive on element 1)

#plot a typical raw signal (transmit and receive on element 1)
(fig, (ax1, ax2)) = plt.subplots(1, 2)
ax1.plot(time * 1e6, fmc_data[0,0,:])
ax1.set_xlabel('Time (us)')
ax1.set_ylabel('Element')
#plot slice through raw data (transmit on element 1, receive on all)
ax2.imshow(fmc_data[:,0,:], extent=[np.min(time) * 1e6, np.max(time) * 1e6, 1, len(element_positions)], aspect = 'auto')
ax2.set_title('Raw data')
ax2.set_xlabel('Time (us)')
ax2.set_ylabel('Element')


# Filter the data
dt = time[1] - time[0]
fft_pts = time.shape[0]
df = 1 / (fft_pts * dt)


spectra = np.fft.fft(fmc_data, fft_pts)
spectra = spectra[0 : int(fft_pts / 2), :, :]
f = np.arange(spectra.shape[2]) * df
f_max = np.max(f)

#define a filter
filter_function = nde.fn_hanning_band_pass(f.shape[0], 0, 1, 0.08, 0.12)

#filter all the signals
filtered_spectra = spectra * filter_function[None, None,:]

#inverse FFT
filtered_signals = np.fft.ifft(filtered_spectra, fft_pts)

resolution = 50

I = np.zeros((resolution, resolution))

x_grid = np.linspace(-grid_size_x / 2, grid_size_x / 2, resolution)  
y_grid = np.linspace(0, grid_size_y, resolution)  



(fig, (ax1)) = plt.subplots(1)
ax1.plot(time * 1e6, filtered_signals[0,0,:])
ax1.set_xlabel('Time (us)')
ax1.set_ylabel('Element')
#plot slice through raw data (transmit on element 1, receive on all)

