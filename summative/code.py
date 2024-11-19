import numpy as np
import matplotlib.pyplot as plt
import NDE_functions as nde
from NDE_functions import fn_simulate_data_weld_v5
from scipy.signal import hilbert  # Add this import

# Initial test parameters
no_elements = 64
element_pitch = 1.5e-3  # 1.5 mm
element_width = 1.45e-3  # 1.45 mm
centre_freq = 2.0e6      # 5 MHz
first_element_position = 1e-3  # 10 mm from weld edge
scan_position = 10.0e-3    # Start of scan
velocity = 6020.0      # m/s

# Get simulated data for points
time_data, time, element_position = fn_simulate_data_weld_v5(
    '_CRACK_BOTTOM', 
    scan_position,
    no_elements,
    element_pitch,
    element_width,
    first_element_position,
    centre_freq
)

# Filtering
dt = time[1] - time[0]
fft_pts = time.shape[0]
spectrum = np.fft.fft(time_data, fft_pts)
spectrum = spectrum[:, :, 0:int(fft_pts/2)]

df = 1 / (fft_pts * dt)
f = np.arange(spectrum.shape[2]) * df
f_max = np.max(f)


f1, f2, f3, f4 = 0.1e7/f_max, 0.2e7/f_max, 0.2e7/f_max, 0.3e7/f_max

# Create bandpass filter (adjust frequencies as needed)
filter_function = nde.fn_hanning_band_pass(f.shape[0],f1, f2, f3, f4)


# Apply filter
filtered_spectrum = spectrum * filter_function[None, None, :]
invspec = np.fft.ifft(filtered_spectrum, fft_pts)

# Apply Hilbert transform after filtering
analytic_signal = hilbert(np.real(invspec), axis=2)

# Update the plotting section to show Hilbert transform result
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, figsize=(10, 16))  # Added one more subplot

# Plot 1: Raw signal
ax1.plot(time * 1e6, time_data[0, 0, :])
ax1.set_xlabel('Time (μs)')
ax1.set_ylabel('Amplitude')
ax1.set_title('Raw Signal')

# Plot 2: Spectrum and filter
normalized_spectrum = np.abs(spectrum[0, 0, :]) / np.max(np.abs(spectrum[0, 0, :]))
ax2.plot(f / 1e6, normalized_spectrum, label='Signal Spectrum')
ax2.plot(f / 1e6, filter_function, ':', label='Filter Function')
ax2.set_xlabel('Frequency (MHz)')
ax2.set_ylabel('Normalized Amplitude')
ax2.set_title('Frequency Spectrum and Filter')
ax2.legend()
ax2.set_xlim([0, 10])

# Plot 3: Filtered signal
ax3.plot(time * 1e6, np.real(invspec[0, 0, :]), label='Real Part')
ax3.plot(time * 1e6, np.abs(invspec[0, 0, :]), ':', label='Envelope')
ax3.set_xlabel('Time (μs)')
ax3.set_ylabel('Amplitude')
ax3.set_title('Filtered Signal')
ax3.legend()

# Plot 4: Hilbert transform result
ax4.plot(time * 1e6, np.real(analytic_signal[0, 0, :]), label='Real Part')
ax4.plot(time * 1e6, np.abs(analytic_signal[0, 0, :]), ':', label='Envelope')
ax4.set_xlabel('Time (μs)')
ax4.set_ylabel('Amplitude')
ax4.set_title('Signal After Hilbert Transform')
ax4.legend()

plt.tight_layout()
plt.show()

# TFM Imaging
# [Previous grid setup code remains the same]
# TFM Imaging
# Set up imaging grid (adjusted for weld geometry)
res = 100
grid_size_x = 50e-3
grid_size_y = 70e-3
# Create grid adjusted for weld position
x_grid = np.linspace(-grid_size_x, grid_size_x, res)
y_grid = np.linspace(0, grid_size_y, res)
X, Y = np.meshgrid(x_grid, y_grid)
# Initialize image array
I = np.zeros((res, res), dtype=complex)
# Precompute distances
distances = np.zeros((no_elements, res, res))
for i in range(no_elements):
    distances[i] = np.sqrt((X - element_position[i])**2 + Y**2)
    
    
# Modified TFM focusing using analytic signal
for T in range(no_elements):
    for R in range(no_elements):
        tau = (distances[T] + distances[R]) / velocity
        tau_idx = np.searchsorted(time, tau.flatten())
        tau_idx = np.clip(tau_idx, 0, len(time) - 1)
        G = analytic_signal[T, R, tau_idx].reshape(res, res)  # Using analytic_signal instead of invspec
        I += G

# Convert to dB scale
I_dB = 20 * np.log10(np.abs(I) / np.max(np.abs(I)))

# Plot TFM image
plt.figure(figsize=(8, 6))
plt.imshow(
    I_dB,
    extent=[np.min(x_grid) * 1e3, np.max(x_grid) * 1e3, np.max(y_grid) * 1e3, np.min(y_grid) * 1e3],
    aspect='equal',
    cmap='viridis',
    vmin=-20,
    vmax=0
)
plt.colorbar(label='Amplitude (dB)')
plt.xlabel('X Position (mm)')
plt.ylabel('Y Position (mm)')
plt.title('TFM Image with Hilbert Transform (dB Scale)')
plt.show()