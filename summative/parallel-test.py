import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import NDE_functions as nde
from NDE_functions import fn_simulate_data_weld_v5

def calculate_distances_chunk(params):
    """Calculate distances for a chunk of elements"""
    start_idx, end_idx, element_position, X, Y = params
    distances_chunk = np.zeros((end_idx - start_idx, X.shape[0], X.shape[1]))
    for i in range(start_idx, end_idx):
        distances_chunk[i - start_idx] = np.sqrt((X - element_position[i])**2 + Y**2)
    return start_idx, distances_chunk

def process_tfm_chunk(params):
    """Process TFM focusing for a chunk of transmitter elements"""
    T_start, T_end, no_elements, distances, time, invspec, velocity, res = params
    I_chunk = np.zeros((res, res), dtype=complex)
    
    for T in range(T_start, T_end):
        for R in range(no_elements):
            tau = (distances[T] + distances[R]) / velocity
            tau_idx = np.searchsorted(time, tau.flatten())
            tau_idx = np.clip(tau_idx, 0, len(time) - 1)
            G = invspec[T, R, tau_idx].reshape(res, res)
            I_chunk += G
    
    return I_chunk

def parallel_tfm_imaging(time_data, time, element_position, no_elements, velocity, 
                        first_element_position, grid_size_x=300e-3, grid_size_y=300e-3, 
                        res=50, num_processes=None):
    """Main function for parallel TFM imaging"""
    if num_processes is None:
        num_processes = mp.cpu_count()

    # Set up imaging grid
    x_grid = np.linspace(-grid_size_x/2, grid_size_x/2, res) + first_element_position
    y_grid = np.linspace(0, grid_size_y, res)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Parallel distance calculation
    chunk_size = max(1, no_elements // num_processes)
    chunks = []
    for i in range(0, no_elements, chunk_size):
        end_idx = min(i + chunk_size, no_elements)
        chunks.append((i, end_idx, element_position, X, Y))

    distances = np.zeros((no_elements, res, res))
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for start_idx, chunk_distances in executor.map(calculate_distances_chunk, chunks):
            end_idx = start_idx + chunk_distances.shape[0]
            distances[start_idx:end_idx] = chunk_distances

    # Parallel TFM focusing
    chunk_size = max(1, no_elements // num_processes)
    chunks = []
    for i in range(0, no_elements, chunk_size):
        end_idx = min(i + chunk_size, no_elements)
        chunks.append((i, end_idx, no_elements, distances, time, time_data, velocity, res))

    I = np.zeros((res, res), dtype=complex)
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        chunk_results = list(executor.map(process_tfm_chunk, chunks))
        for chunk_result in chunk_results:
            I += chunk_result

    return I, x_grid, y_grid

# Example usage:
if __name__ == '__main__':
    # Initial test parameters (same as original)
    no_elements = 30
    element_pitch = 0.6e-3
    element_width = 0.5e-3
    centre_freq = 5e6
    first_element_position = 200e-3
    scan_position = 100.0e-3
    velocity = 6020.0

    # Get simulated data
    time_data, time, element_position = fn_simulate_data_weld_v5(
        '_CRACK_MIDDLE', 
        scan_position,
        no_elements,
        element_pitch,
        element_width,
        first_element_position,
        centre_freq
    )

    # Filtering (same as original)
    dt = time[1] - time[0]
    fft_pts = time.shape[0]
    spectrum = np.fft.fft(time_data, fft_pts)
    spectrum = spectrum[:, :, 0:int(fft_pts/2)]
    df = 1 / (fft_pts * dt)
    f = np.arange(spectrum.shape[2]) * df
    f_max = np.max(f)
    f1, f2, f3, f4 = 0.3e7/f_max, 0.5e7/f_max, 0.5e7/f_max, 0.7e7/f_max
    filter_function = nde.fn_hanning_band_pass(f.shape[0], f1, f2, f3, f4)
    filtered_spectrum = spectrum * filter_function[None, None, :]
    invspec = np.fft.ifft(filtered_spectrum, fft_pts)

    # Parallel TFM imaging
    I, x_grid, y_grid = parallel_tfm_imaging(
        invspec, time, element_position, 
        no_elements, velocity, first_element_position
    )

    # Convert to dB scale and plot
    I_dB = 20 * np.log10(np.abs(I) / np.max(np.abs(I)))
    plt.figure(figsize=(8, 6))
    plt.imshow(
        I_dB,
        extent=[np.min(x_grid) * 1e3, np.max(x_grid) * 1e3, 
                np.max(y_grid) * 1e3, np.min(y_grid) * 1e3],
        aspect='auto',
        cmap='viridis',
        vmin=-40,
        vmax=0
    )
    plt.colorbar(label='Amplitude (dB)')
    plt.xlabel('X Position (mm)')
    plt.ylabel('Y Position (mm)')
    plt.title('TFM Image (dB Scale)')
    plt.show()