# fm_synthesis.py

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import librosa

# Waveform generator for sine wave
def sine_wave(frequency, amplitude, duration, sample_rate, modulator=None):
    time_vector = np.linspace(0, duration, int(sample_rate * duration))
    if modulator is not None:
        return amplitude * np.sin(2 * np.pi * frequency * time_vector + modulator)
    return amplitude * np.sin(2 * np.pi * frequency * time_vector)

# FM modulation setup
def fm_modulate(carrier_freq, carrier_amp, modulator_signal, duration, sample_rate):
    time_vector = np.linspace(0, duration, int(sample_rate * duration))
    return carrier_amp * np.sin(2 * np.pi * carrier_freq * time_vector + modulator_signal)

# Compute MFCC
def compute_mfcc(signal, sample_rate, n_mfcc=20):
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

# Objective function template
def compute_objective(params, target_freqs, target_amps, duration, sample_rate, objective_type, target_mfcc_mean=None):
    combined_signal = np.zeros(int(sample_rate * duration))
    
    # Unpack parameters for each oscillator
    freq4, amp4 = params[0], params[1]
    freq3, amp3 = params[2], params[3]
    freq2, amp2 = params[4], params[5]
    freq1, amp1 = params[6], params[7]

    # Generate modulator signals
    mod4 = sine_wave(freq4, amp4, duration, sample_rate)
    mod3 = fm_modulate(freq3, amp3, mod4, duration, sample_rate)
    mod2 = fm_modulate(freq2, amp2, mod3, duration, sample_rate)
    combined_signal += fm_modulate(freq1, amp1, mod2, duration, sample_rate)
    
    # Normalize combined signal
    max_val = np.max(np.abs(combined_signal))
    if max_val > 0:
        combined_signal /= max_val
    
    # Compute FFT
    fft_result = np.abs(np.fft.fft(combined_signal))
    fft_freqs = np.fft.fftfreq(len(fft_result), 1/sample_rate)
    
    # Compute the target spectrum
    target_spectrum = np.zeros_like(fft_result)
    for target_freq, target_amp in zip(target_freqs, target_amps):
        closest_index = np.argmin(np.abs(fft_freqs - target_freq))
        target_spectrum[closest_index] = target_amp
    
    epsilon = 1e-10  # Small value to avoid log(0) and division by zero
    
    if objective_type == 'itakura_saito':
        is_distance = np.sum((target_spectrum + epsilon) / (fft_result + epsilon) - np.log((target_spectrum + epsilon) / (fft_result + epsilon)) - 1)
        return is_distance
    
    elif objective_type == 'spectral_convergence':
        spectral_convergence = np.linalg.norm(np.abs(fft_result) - np.abs(target_spectrum)) / np.linalg.norm(np.abs(target_spectrum))
        return spectral_convergence
    
    elif objective_type == 'cosine_similarity':
        cosine_sim = cosine(np.abs(fft_result), np.abs(target_spectrum))
        return cosine_sim
    
    elif objective_type == 'euclidean_distance':
        euclidean_distance = np.linalg.norm(np.abs(fft_result) - np.abs(target_spectrum))
        return euclidean_distance
    
    elif objective_type == 'manhattan_distance':
        manhattan_distance = np.sum(np.abs(np.abs(fft_result) - np.abs(target_spectrum)))
        return manhattan_distance
    
    elif objective_type == 'kullback_leibler_divergence':
        kl_divergence = np.sum(target_spectrum * np.log((target_spectrum + epsilon) / (fft_result + epsilon)))
        return kl_divergence

    elif objective_type == 'pearson_correlation_coefficient':
        pearson_corr, _ = pearsonr(np.abs(fft_result), target_spectrum)
        return 1 - pearson_corr  # We subtract from 1 because we want to minimize this objective function
    
    elif objective_type == 'mfcc_distance':
        generated_mfcc_mean = compute_mfcc(combined_signal, sample_rate)
        mfcc_distance = np.linalg.norm(generated_mfcc_mean - target_mfcc_mean)
        return mfcc_distance
