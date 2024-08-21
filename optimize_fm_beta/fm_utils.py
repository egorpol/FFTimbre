import matplotlib.pyplot as plt
import numpy as np

def plot_results(combined_signal, frequencies, amplitudes, error_history, sample_rate):
    # Time domain plot
    plt.figure(figsize=(12, 6))
    plt.plot(combined_signal[:1000])
    plt.title('Optimized Combined Signal in Time Domain')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.show()

    # Frequency domain plot
    fft_result = np.fft.fft(combined_signal)
    fft_freqs = np.fft.fftfreq(len(fft_result), 1/sample_rate)
    fft_result_np = np.abs(fft_result)
    fft_result_np /= np.max(fft_result_np)

    plt.figure(figsize=(12, 6))
    plt.plot(fft_freqs[:len(fft_freqs)//2], fft_result_np[:len(fft_result_np)//2], label='Optimized Spectrum')
    plt.scatter(frequencies, amplitudes, color='red', label='Target Spectrum', zorder=5)
    plt.title('Frequency Spectrum of Optimized Combined Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

    # Error history plot
    plt.figure(figsize=(12, 6))
    plt.plot(error_history)
    plt.title('Error Rate Progression')
    plt.xlabel('Generation')
    plt.ylabel('Distance')
    plt.grid(True)
    plt.show()

def print_optimal_params(optimal_params):
    for i, (freq, amp) in enumerate(zip(optimal_params[0::2], optimal_params[1::2])):
        print(f"Modulator {4-i}:\n")
        print(f"    Frequency: {freq}")
        print(f"    Amplitude: {amp}\n")
