import numpy as np
import matplotlib.pyplot as plt
import NDE_functions as nde
from NDE_functions import fn_simulate_data_weld_v5
from scipy.signal import hilbert
import os

# Create folder for saving images if it doesn't exist
save_folder = 'weld_scan_images'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Keep your current optimized parameters
no_elements = 64
element_pitch = 1.5e-3
element_width = 1.45e-3
centre_freq = 2.0e6  
first_element_position = 1e-3
velocity = 6020.0

# Set up scanning parameters
mode = 'ey21933'  # Replace with your username
scan_positions = np.arange(0, 0.300, 0.001)  # 0 to 300mm in 5mm steps

# Loop through scan positions
for i, scan_position in enumerate(scan_positions):
    print(f"Processing position {scan_position*1e3:.1f}mm ({i+1}/{len(scan_positions)})")
    
    # Get simulated data
    time_data, time, element_position = fn_simulate_data_weld_v5(
        mode, 
        scan_position,
        no_elements,
        element_pitch,
        element_width,
        first_element_position,
        centre_freq
    )

    # Your existing filtering code
    dt = time[1] - time[0]
    fft_pts = time.shape[0]
    spectrum = np.fft.fft(time_data, fft_pts)
    spectrum = spectrum[:, :, 0:int(fft_pts/2)]

    df = 1 / (fft_pts * dt)
    f = np.arange(spectrum.shape[2]) * df
    f_max = np.max(f)

    f1, f2, f3, f4 = 0.1e7/f_max, 0.2e7/f_max, 0.2e7/f_max, 0.3e7/f_max
    filter_function = nde.fn_hanning_band_pass(f.shape[0],f1, f2, f3, f4)

    filtered_spectrum = spectrum * filter_function[None, None, :]
    invspec = np.fft.ifft(filtered_spectrum, fft_pts)
    analytic_signal = hilbert(np.real(invspec), axis=2)

    # TFM imaging
    res = 100
    grid_size_x = 50e-3
    grid_size_y = 70e-3
    x_grid = np.linspace(-grid_size_x, grid_size_x, res)
    y_grid = np.linspace(0, grid_size_y, res)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    I = np.zeros((res, res), dtype=complex)
    distances = np.zeros((no_elements, res, res))
    
    for i_elem in range(no_elements):
        distances[i_elem] = np.sqrt((X - element_position[i_elem])**2 + Y**2)
    
    for T in range(no_elements):
        for R in range(no_elements):
            tau = (distances[T] + distances[R]) / velocity
            tau_idx = np.searchsorted(time, tau.flatten())
            tau_idx = np.clip(tau_idx, 0, len(time) - 1)
            G = analytic_signal[T, R, tau_idx].reshape(res, res)
            I += G

    # Convert to dB
    I_dB = 20 * np.log10(np.abs(I) / np.max(np.abs(I)))

    # Create figure for this position
    plt.figure(figsize=(8, 6))
    plt.imshow(
        I_dB,
        extent=[np.min(x_grid) * 1e3, np.max(x_grid) * 1e3, np.max(y_grid) * 1e3, np.min(y_grid) * 1e3],
        aspect='equal',
        cmap='viridis',
        vmin=-40,
        vmax=0
    )
    plt.colorbar(label='Amplitude (dB)')
    plt.xlabel('X Position (mm)')
    plt.ylabel('Y Position (mm)')
    plt.title(f'TFM Image - Position: {scan_position*1e3:.1f}mm')

    # Add -6dB contour for sizing
    plt.contour(
        x_grid * 1e3,
        y_grid * 1e3,
        I_dB,
        levels=[-6],
        colors='r',
        linewidths=2
    )

    # Save figure
    plt.savefig(os.path.join(save_folder, f'scan_position_{scan_position*1e3:.1f}mm.png'))
    plt.close()  # Close the figure to free memory

print(f"Scan complete. Images saved in folder: {save_folder}")