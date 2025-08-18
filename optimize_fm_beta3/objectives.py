import numpy as np
import librosa
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from synthesis import sine_wave

# --- Feature Extraction ---
def compute_mfcc(signal, sample_rate, n_mfcc=20):
    """Computes the mean of MFCCs for a given signal."""
    mfccs = librosa.feature.mfcc(y=signal.astype(np.float32), sr=sample_rate, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1)

def compute_fft(signal, sample_rate):
    """Computes the FFT magnitude and corresponding frequencies."""
    fft_result = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(len(fft_result), 1 / sample_rate)
    fft_magnitude = np.abs(fft_result)
    return fft_magnitude, fft_freqs

# --- Distance Metrics ---
# These now take pre-computed features as input
def mfcc_distance(gen_mfcc, target_mfcc):
    return np.linalg.norm(gen_mfcc - target_mfcc)

def cosine_similarity_dist(gen_spec, target_spec):
    return cosine(gen_spec, target_spec)

def euclidean_distance(gen_spec, target_spec):
    return np.linalg.norm(gen_spec - target_spec)

def pearson_correlation_dist(gen_spec, target_spec):
    corr, _ = pearsonr(gen_spec, target_spec)
    # Return 1 - corr because optimizers minimize, and we want to maximize correlation
    return 1 - corr

def itakura_saito_dist(gen_spec, target_spec):
    """
    Computes the Itakura-Saito divergence between two spectra.
    Typically used with power spectra, but applied here to magnitude for consistency.
    """
    epsilon = 1e-10  # Small constant to avoid division by zero and log(0)
    
    # Ensure spectra are positive
    gen_spec = np.maximum(gen_spec, epsilon)
    target_spec = np.maximum(target_spec, epsilon)
    
    # IS divergence formula: sum( target/gen - log(target/gen) - 1 )
    ratio = target_spec / gen_spec
    log_ratio = np.log(ratio)
    
    # Sum over all frequency bins
    is_distance = np.sum(ratio - log_ratio - 1)
    
    return is_distance

# +++ UPDATE THE DICTIONARIES +++
METRIC_FUNCTIONS = {
    'mfcc_distance': mfcc_distance,
    'cosine_similarity': cosine_similarity_dist,
    'euclidean_distance': euclidean_distance,
    'pearson_correlation_coefficient': pearson_correlation_dist,
    'itakura_saito': itakura_saito_dist, # Add the new function here
}

METRIC_TYPE = {
    'mfcc_distance': 'mfcc',
    'cosine_similarity': 'spectrum',
    'euclidean_distance': 'spectrum',
    'pearson_correlation_coefficient': 'spectrum',
    'itakura_saito': 'spectrum', # Specify that it operates on the spectrum
}

# --- Target Preparation ---
def prepare_target(frequencies, amplitudes, duration, sample_rate, objective_type):
    """Pre-computes the target data (MFCC or Spectrum) to avoid re-calculation."""
    print("Generating and analyzing target signal...")
    target_signal = np.zeros(int(sample_rate * duration))
    for freq, amp in zip(frequencies, amplitudes):
        target_signal += sine_wave(freq, amp, duration, sample_rate)
    
    max_val = np.max(np.abs(target_signal))
    if max_val > 0:
        target_signal /= max_val

    feature_type = METRIC_TYPE.get(objective_type)
    if feature_type is None:
        raise ValueError(f"Unknown objective type: {objective_type}. Not found in METRIC_TYPE dictionary.")

    if feature_type == 'mfcc':
        return compute_mfcc(target_signal, sample_rate)
    
    elif feature_type == 'spectrum':
        # For spectral distances, we create a sparse target spectrum
        full_target_magnitude, fft_freqs = compute_fft(target_signal, sample_rate)
        target_spectrum_sparse = np.zeros_like(full_target_magnitude)
        
        # Normalize target amplitudes to match the scale of the FFT magnitude of a normalized signal
        amplitudes_normalized = amplitudes / np.max(amplitudes)

        for freq, amp in zip(frequencies, amplitudes_normalized):
            idx = np.argmin(np.abs(fft_freqs - freq))
            # The amplitude in the FFT is proportional to N*amp/2.
            # However, since we compare to the FFT of a normalized signal,
            # using the direct [0,1] amplitudes is a valid proxy.
            target_spectrum_sparse[idx] = amp

        return target_spectrum_sparse