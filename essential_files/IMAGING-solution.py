# -*- coding: utf-8 -*-
import NDE_functions as nde
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# array parameters
no_elements = 32
element_pitch = 0.6e-3
element_width = 0.5e-3
centre_frequency = 5e6
time_pts = 1000

# imaging parameters
grid_size_x = 50e-3  # 50 mm converted to meters
grid_size_y = 50e-3  # 50 mm converted to meters
grid_pixel_size = 0.5e-3  # 0.5 mm converted to meters

# wave parameters
velocity = 6e3  # velocity 

# generate raw data
(fmc_data, time, element_positions) = nde.fn_simulate_data_ex5_v2(
    no_elements, element_pitch, element_width, centre_frequency, time_pts
)

(fig, (ax1, ax2)) = plt.subplots(1, 2)
ax1.plot(time * 1e6, fmc_data[0, 0, :])
ax1.set_xlabel('Time (us)')
ax1.set_ylabel('Element')

ax2.imshow(
    fmc_data[:, 0, :],
    extent=[np.min(time) * 1e6, np.max(time) * 1e6, 1, len(element_positions)],
    aspect='auto',
)
ax2.set_title('Raw data')
ax2.set_xlabel('Time (us)')
ax2.set_ylabel('Intensity')

# Filtering
dt = time[1] - time[0]
fft_pts = time.shape[0]
spectrum = np.fft.fft(fmc_data, fft_pts)
spectrum = spectrum[0 : int(fft_pts / 2), :, :]

df = 1 / (fft_pts * dt)
f = np.arange(spectrum.shape[2]) * df
f_max = np.max(f)

filter_function = nde.fn_hanning_band_pass(f.shape[0], 0.35e7/f_max, 0.4e7/f_max, 0.6e7/f_max, 0.65e7/f_max)
multspec = spectrum * filter_function[None, None, :]
invspec = np.fft.ifft(multspec, fft_pts)

# Create figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(10, 12))

# Plot 1: Raw signal in time domain
ax1.plot(time * 1e6, fmc_data[0, 0, :])
ax1.set_xlabel('Time (μs)')
ax1.set_ylabel('Amplitude')
ax1.set_title('Raw Signal')

# Plot 2: Frequency spectrum and filter
freq = f  # Use the frequency array we calculated
normalized_spectrum = np.abs(spectrum[0, 0, :]) / np.max(np.abs(spectrum[0, 0, :]))
ax2.plot(freq / 1e6, normalized_spectrum, label='Signal Spectrum')
ax2.plot(freq / 1e6, filter_function, ':', label='Filter Function')
ax2.set_xlabel('Frequency (MHz)')
ax2.set_ylabel('Normalized Amplitude')
ax2.set_title('Frequency Spectrum and Filter')
ax2.legend()
ax2.set_xlim([0, 10])  # Show up to 10 MHz

# Plot 3: Filtered signal
ax3.plot(time * 1e6, np.real(invspec[0, 0, :]), label='Real Part')
ax3.plot(time * 1e6, np.abs(invspec[0, 0, :]), ':', label='Envelope')
ax3.set_xlabel('Time (μs)')
ax3.set_ylabel('Amplitude')
ax3.set_title('Filtered Signal')
ax3.legend()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


res = 200
# Use the complex types 
x_grid = np.linspace(-grid_size_x / 2, grid_size_x / 2, res)
y_grid = np.linspace(0, grid_size_y, res)
X, Y = np.meshgrid(x_grid, y_grid)

# Initialize output array
I = np.zeros((res, res), dtype=complex)

# Get number of elements
no_elements = len(element_positions)

# Precompute distances for each element to all grid points
distances = np.zeros((no_elements, res, res))
for i in range(no_elements):
    distances[i] = np.sqrt((X - element_positions[i])**2 + Y**2)

# Main loop over transmitter and receiver elements
for T in range(no_elements):
    for R in range(no_elements):
        # Calculate total path length (transmitter to point to receiver)
        tau = (distances[T] + distances[R]) / velocity
        
        # Find nearest time index for each point (vectorized)
        tau_idx = np.searchsorted(time, tau.flatten())
        tau_idx = np.clip(tau_idx, 0, len(time) - 1)
        
        # Get corresponding G values
        G = invspec[T, R, tau_idx].reshape(res, res)
        
        # Add to image
        I += G

# Convert to dB scale
I_dB = 20 * np.log10(np.abs(I) / np.max(np.abs(I)))

plt.figure(figsize=(8, 6))
plt.imshow(
    I_dB,
    extent=[np.min(x_grid) * 1e3, np.max(x_grid) * 1e3, np.max(y_grid) * 1e3, np.min(y_grid) * 1e3],
    aspect="auto",
    cmap="viridis",
    vmin=-40,
    vmax=0,)
plt.colorbar(label='Amplitude (dB)')
plt.xlabel('X Position (mm)')
plt.ylabel('Y Position (mm)')
plt.title('TFM Image (dB Scale)')
plt.show()