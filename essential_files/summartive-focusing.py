import numpy as np
import matplotlib.pyplot as plt
from NDE_functions import fn_simulate_data_weld_v5
import NDE_functions as nde

# Array parameters
no_elements = 32
element_pitch = 0.6e-3  # 0.6 mm
element_width = 0.5e-3  # 0.5 mm
centre_freq = 5e6      # 5 MHz
first_element_position = 10e-3  # 10 mm from weld edge
scan_position = 0.0    
velocity = 6020.0      # m/s

# Calculate wavelength for focal point calculations
wavelength = velocity / centre_freq
k = 2 * np.pi / wavelength

# Define more focal points along the 45° fusion face
# Create 9 evenly spaced points along the fusion face
depths = np.linspace(5e-3, 35e-3, 9)  # From 5mm to 35mm depth
focal_points = np.array([[d, d] for d in depths]) + np.array([first_element_position, 0])

# Get simulated data
time_data, time, element_position = fn_simulate_data_weld_v5(
    '_POINTS_NO_NOISE', 
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
spectrum = spectrum[0 : int(fft_pts / 2), :, :]
df = 1 / (fft_pts * dt)
f = np.arange(spectrum.shape[2]) * df
f_max = np.max(f)
filter_function = nde.fn_hanning_band_pass(f.shape[0], 0.35e7/f_max, 0.4e7/f_max, 0.6e7/f_max, 0.65e7/f_max)
multspec = spectrum * filter_function[None, None, :]
invspec = np.fft.ifft(multspec, fft_pts)

# Create plots for each focal point
plt.figure(figsize=(15, 15))
for idx, focal_point in enumerate(focal_points, 1):
    # Calculate focusing delays for this focal point
    delays = np.zeros(no_elements)
    ref_dist = np.sqrt(np.sum((focal_point - np.array([element_position[0], 0]))**2))
    
    for i in range(no_elements):
        dist = np.sqrt(np.sum((focal_point - np.array([element_position[i], 0]))**2))
        delays[i] = (dist - ref_dist) / velocity

    # Set up imaging grid
    res = 200
    grid_size_x = 40e-3
    grid_size_y = 40e-3
    
    x_grid = np.linspace(0, grid_size_x, res) + first_element_position
    y_grid = np.linspace(0, grid_size_y, res)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Initialize and compute TFM image
    I = np.zeros((res, res), dtype=complex)
    distances = np.zeros((no_elements, res, res))
    
    for i in range(no_elements):
        distances[i] = np.sqrt((X - element_position[i])**2 + Y**2)

    # TFM focusing with delays
    for T in range(no_elements):
        for R in range(no_elements):
            total_delay = delays[T] + delays[R]
            tau = (distances[T] + distances[R]) / velocity - total_delay
            tau_idx = np.searchsorted(time, tau.flatten())
            tau_idx = np.clip(tau_idx, 0, len(time) - 1)
            G = invspec[T, R, tau_idx].reshape(res, res)
            I += G

    # Plot subfigure for this focal point
    plt.subplot(3, 3, idx)
    I_dB = 20 * np.log10(np.abs(I) / np.max(np.abs(I)))
    
    plt.imshow(
        I_dB,
        extent=[np.min(x_grid) * 1e3, np.max(x_grid) * 1e3, np.max(y_grid) * 1e3, np.min(y_grid) * 1e3],
        aspect='equal',
        cmap='viridis',
        vmin=-40,
        vmax=0
    )
    
    # Plot focal point and fusion face line
    plt.plot(focal_point[0]*1e3, focal_point[1]*1e3, 'r+', markersize=10, label='Focal point')
    x_face = np.linspace(0, 40, 100)
    y_face = x_face
    plt.plot((x_face + first_element_position*1e3), y_face, 'r--', alpha=0.5, label='Fusion face')
    
    plt.colorbar(label='Amplitude (dB)')
    plt.xlabel('X Position (mm)')
    plt.ylabel('Y Position (mm)')
    plt.title(f'Focus at ({focal_point[0]*1e3:.1f}, {focal_point[1]*1e3:.1f}) mm')
    if idx == 1:
        plt.legend()

plt.tight_layout()
plt.show()