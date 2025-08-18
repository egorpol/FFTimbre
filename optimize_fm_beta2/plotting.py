# plotting.py
import numpy as np
import matplotlib.pyplot as plt

def plot_time_domain_signal(signal, sample_rate, sample_count=1000):
    plt.figure(figsize=(12, 6))
    # This is the corrected line:
    time_axis = np.arange(len(signal)) / sample_rate 
    plt.plot(time_axis[:sample_count], signal[:sample_count])
    plt.title('Optimized Signal (Time Domain)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

def plot_frequency_domain_signal(fft_freqs, fft_magnitude, target_freqs, target_amps):
    plt.figure(figsize=(12, 6))
    
    # --- START OF FIX ---
    # Normalize both spectra to a [0, 1] range for visual comparison
    
    # Normalize the generated spectrum's magnitude
    max_fft_mag = np.max(fft_magnitude)
    if max_fft_mag > 0:
        fft_magnitude_normalized = fft_magnitude / max_fft_mag
    else:
        fft_magnitude_normalized = fft_magnitude

    # Normalize the target amplitudes (they should already be [0,1], but this is safe)
    max_target_amp = np.max(target_amps)
    if max_target_amp > 0:
        target_amps_normalized = target_amps / max_target_amp
    else:
        target_amps_normalized = target_amps
    # --- END OF FIX ---

    # Plot only the positive frequencies using the NORMALIZED data
    positive_mask = fft_freqs >= 0
    plt.plot(fft_freqs[positive_mask], fft_magnitude_normalized[positive_mask], label='Optimized Spectrum (Normalized)')
    plt.scatter(target_freqs, target_amps_normalized, color='red', label='Target Partials (Normalized)', zorder=5)
    
    plt.title('Frequency Spectrum Comparison')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized Magnitude')
    plt.xlim(0, max(target_freqs) * 1.2)
    plt.ylim(0, 1.1) # Set y-axis limit for normalized data
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_error_history(error_history):
    if not error_history:
        print("No error history to plot (possibly due to optimizer choice).")
        return
    plt.figure(figsize=(12, 6))
    plt.plot(error_history)
    plt.title('Optimization Error History')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value (Distance)')
    plt.grid(True)
    plt.yscale('log') # Log scale is often better for viewing convergence
    plt.show()