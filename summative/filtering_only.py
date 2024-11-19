import numpy as np
import matplotlib.pyplot as plt
import NDE_functions as nde
from NDE_functions import fn_simulate_data_weld_v5
from scipy import signal

def apply_ndt_filtering(time_data, time, centre_freq):
    """
    Apply FIR and IIR filtering to NDT data with windowing for sidelobe suppression
    """
    dt = time[1] - time[0]
    fs = 1/dt
    nyq = fs/2
    
    # Print debugging information
    print(f"Sampling frequency: {fs/1e6:.2f} MHz")
    print(f"Nyquist frequency: {nyq/1e6:.2f} MHz")
    print(f"Center frequency: {centre_freq/1e6:.2f} MHz")
    
    # Design FIR filter with Kaiser window
    numtaps = 151
    beta = 6.0  # Kaiser window parameter (higher beta = more sidelobe suppression)
    fir_coeff = signal.firwin(numtaps, 
                            [centre_freq-1e6, centre_freq+1e6], 
                            window=('kaiser', beta),
                            fs=fs,
                            pass_zero=False)
    
    # Design IIR (Butterworth) filter with higher order
    order = 8  # Increased order for sharper cutoff
    wn = np.array([centre_freq-1e6, centre_freq+1e6])/nyq
    wn = np.clip(wn, 0, 0.99)
    b, a = signal.butter(order, wn, btype='bandpass')
    
    # Initialize output arrays
    fir_filtered = np.zeros_like(time_data, dtype=complex)
    iir_filtered = np.zeros_like(time_data, dtype=complex)
    
    # Apply filters to each A-scan
    for tx in range(time_data.shape[0]):
        for rx in range(time_data.shape[1]):
            # Apply window to the data before filtering
            windowed_data = time_data[tx, rx, :] * signal.windows.hann(time_data.shape[2])
            
            # FIR filtering
            fir_filtered[tx, rx, :] = signal.filtfilt(fir_coeff, [1], 
                                                    windowed_data)
            
            # IIR filtering
            iir_filtered[tx, rx, :] = signal.filtfilt(b, a, 
                                                    windowed_data)
    
    return fir_filtered, iir_filtered

# Initial test parameters
no_elements = 32
element_pitch = 0.6e-3  # 0.6 mm
element_width = 0.5e-3  # 0.5 mm
centre_freq = 5e6      # 5 MHz
first_element_position = 10e-3  # 10 mm from weld edge
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

# Apply filtering
fir_filtered, iir_filtered = apply_ndt_filtering(time_data, time, centre_freq)


# Plotting
fig, axes = plt.subplots(5, 1, figsize=(12, 20))

# Plot 1: Raw signal
axes[0].plot(time * 1e6, time_data[0, 0, :])
axes[0].set_title('Raw Signal')
axes[0].set_xlabel('Time (μs)')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True)

# Plot 2: Frequency response
f_response = np.fft.fftfreq(len(time), time[1]-time[0])
raw_spectrum = np.fft.fft(time_data[0, 0, :])
fir_spectrum = np.fft.fft(fir_filtered[0, 0, :])
iir_spectrum = np.fft.fft(iir_filtered[0, 0, :])


axes[1].plot(f_response[:len(f_response)//2]/1e6, 
            np.abs(fir_spectrum[:len(f_response)//2])/np.max(np.abs(fir_spectrum)), 
            label='FIR')
axes[1].plot(f_response[:len(f_response)//2]/1e6, 
            np.abs(iir_spectrum[:len(f_response)//2])/np.max(np.abs(iir_spectrum)), 
            label='IIR')
axes[1].set_title('Frequency Spectrum Comparison')
axes[1].set_xlabel('Frequency (MHz)')
axes[1].set_ylabel('Normalized Amplitude')
axes[1].legend()
axes[1].set_xlim([0, 10])
axes[1].grid(True)

# Plot 3: FIR filtered signal
axes[2].plot(time * 1e6, np.real(fir_filtered[0, 0, :]), label='Signal')
axes[2].plot(time * 1e6, np.abs(fir_filtered[0, 0, :]), ':', label='Envelope')
axes[2].set_title('FIR Filtered Signal with Envelope')
axes[2].set_xlabel('Time (μs)')
axes[2].set_ylabel('Amplitude')
axes[2].legend()
axes[2].grid(True)

# Plot 4: IIR filtered signal
axes[3].plot(time * 1e6, np.real(iir_filtered[0, 0, :]), label='Signal')
axes[3].plot(time * 1e6, np.abs(fir_filtered[0, 0, :]), ':', label='Envelope')
axes[3].set_title('IIR Filtered Signal with Envelope')
axes[3].set_xlabel('Time (μs)')
axes[3].set_ylabel('Amplitude')
axes[3].legend()
axes[3].grid(True)

# Plot 5: Envelope comparison
axes[4].plot(time * 1e6, np.abs(fir_filtered[0, 0, :]), label='FIR Envelope')
axes[4].plot(time * 1e6, np.abs(iir_filtered[0, 0, :]), label='IIR Envelope')
axes[4].set_title('Envelope Comparison')
axes[4].set_xlabel('Time (μs)')
axes[4].set_ylabel('Amplitude')
axes[4].legend()
axes[4].grid(True)

plt.tight_layout()
plt.show()

# TFM processing remains the same as before...
# TFM Imaging setup
res = 1000
grid_size_x = 50e-3
grid_size_y = 40e-3
x_grid = np.linspace(-grid_size_x/2, grid_size_x/2, res) + first_element_position
y_grid = np.linspace(0, grid_size_y, res)
X, Y = np.meshgrid(x_grid, y_grid)

# Initialize image arrays for both FIR and IIR results
I_fir = np.zeros((res, res), dtype=complex)
I_iir = np.zeros((res, res), dtype=complex)

# Precompute distances
distances = np.zeros((no_elements, res, res))
for i in range(no_elements):
    distances[i] = np.sqrt((X - element_position[i])**2 + Y**2)

# TFM focusing using both filtered datasets
for T in range(no_elements):
    for R in range(no_elements):
        tau = (distances[T] + distances[R]) / velocity
        tau_idx = np.searchsorted(time, tau.flatten())
        tau_idx = np.clip(tau_idx, 0, len(time) - 1)
        
        # FIR TFM
        G_fir = fir_filtered[T, R, tau_idx].reshape(res, res)
        I_fir += G_fir
        
        # IIR TFM
        G_iir = iir_filtered[T, R, tau_idx].reshape(res, res)
        I_iir += G_iir

# Convert to dB scale
I_fir_dB = 20 * np.log10(np.abs(I_fir) / np.max(np.abs(I_fir)))
I_iir_dB = 20 * np.log10(np.abs(I_iir) / np.max(np.abs(I_iir)))

# Create figure for TFM images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# FIR TFM image
im1 = ax1.imshow(
    I_fir_dB,
    extent=[np.min(x_grid)*1e3, np.max(x_grid)*1e3, 
            np.max(y_grid)*1e3, np.min(y_grid)*1e3],
    aspect='auto',
    cmap='viridis',
    vmin=-40,
    vmax=0
)
ax1.set_title('TFM Image (FIR Filtered)')
ax1.set_xlabel('X Position (mm)')
ax1.set_ylabel('Y Position (mm)')
plt.colorbar(im1, ax=ax1, label='Amplitude (dB)')

# IIR TFM image
im2 = ax2.imshow(
    I_iir_dB,
    extent=[np.min(x_grid)*1e3, np.max(x_grid)*1e3, 
            np.max(y_grid)*1e3, np.min(y_grid)*1e3],
    aspect='auto',
    cmap='viridis',
    vmin=-40,
    vmax=0
)
ax2.set_title('TFM Image (IIR Filtered)')
ax2.set_xlabel('X Position (mm)')
ax2.set_ylabel('Y Position (mm)')
plt.colorbar(im2, ax=ax2, label='Amplitude (dB)')

plt.tight_layout()
plt.show()

# Optional: Calculate and display image quality metrics
def calculate_image_metrics(image_db):
    """Calculate image quality metrics"""
    # SNR estimation (using top 1% of pixels as signal)
    flat_img = image_db.flatten()
    signal_threshold = np.percentile(flat_img, 99)
    signal = flat_img[flat_img > signal_threshold].mean()
    noise = flat_img[flat_img <= signal_threshold].std()
    snr = signal / noise if noise != 0 else float('inf')
    
    # Dynamic range
    dynamic_range = np.max(image_db) - np.min(image_db)
    
    return snr, dynamic_range

# Calculate metrics for both images
snr_fir, dr_fir = calculate_image_metrics(I_fir_dB)
snr_iir, dr_iir = calculate_image_metrics(I_iir_dB)

print("\nImage Quality Metrics:")
print(f"FIR Filter - SNR: {snr_fir:.2f}, Dynamic Range: {dr_fir:.2f} dB")
print(f"IIR Filter - SNR: {snr_iir:.2f}, Dynamic Range: {dr_iir:.2f} dB")