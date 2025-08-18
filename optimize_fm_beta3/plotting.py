# plotting.py
import numpy as np
import matplotlib.pyplot as plt

def plot_time_domain_signal(signal, sample_rate, sample_count=1000):
    """Plots the signal in the time domain."""
    plt.figure(figsize=(12, 6))
    time_axis = np.arange(len(signal)) / sample_rate 
    plt.plot(time_axis[:sample_count], signal[:sample_count])
    plt.title('Optimized Signal (Time Domain)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

def plot_frequency_domain_signal(fft_freqs, fft_magnitude, target_freqs, target_amps, xlim_max=None):
    """
    Plots the frequency spectrum comparison with a customizable x-axis limit.

    Args:
        fft_freqs (np.ndarray): Frequencies from the FFT of the generated signal.
        fft_magnitude (np.ndarray): Magnitudes from the FFT of the generated signal.
        target_freqs (np.ndarray): Frequencies of the target partials.
        target_amps (np.ndarray): Amplitudes of the target partials.
        xlim_max (float, optional): The maximum frequency (in Hz) to display on the x-axis.
                                     If None, the limit is set automatically based on target frequencies.
                                     Defaults to None.
    """
    plt.figure(figsize=(12, 6))
    
    # Normalize both spectra to a [0, 1] range for visual comparison
    max_fft_mag = np.max(fft_magnitude)
    if max_fft_mag > 0:
        fft_magnitude_normalized = fft_magnitude / max_fft_mag
    else:
        fft_magnitude_normalized = fft_magnitude

    max_target_amp = np.max(target_amps)
    if max_target_amp > 0:
        target_amps_normalized = target_amps / max_target_amp
    else:
        target_amps_normalized = target_amps

    # Plot using the positive-frequency rFFT arrays provided
    plt.plot(fft_freqs, fft_magnitude_normalized, label='Optimized Spectrum (Normalized)')
    plt.scatter(target_freqs, target_amps_normalized, color='red', label='Target Partials (Normalized)', zorder=5)
    
    plt.title('Frequency Spectrum Comparison')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized Magnitude')
    
    # --- START OF FIX: Customizable x-axis limit ---
    if xlim_max is not None and xlim_max > 0:
        plt.xlim(0, xlim_max)
        print(f"Using custom x-axis limit for plot: {xlim_max} Hz")
    else:
        # Default automatic behavior
        plt.xlim(0, max(target_freqs) * 1.2)
    # --- END OF FIX ---
        
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_error_history(error_history):
    """Plots the optimization error history."""
    if not error_history:
        print("No error history to plot (possibly due to optimizer choice).")
        return
    plt.figure(figsize=(12, 6))
    plt.plot(error_history)
    plt.title('Optimization Error History')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value (Distance)')
    plt.grid(True)
    plt.yscale('log')
    plt.show()