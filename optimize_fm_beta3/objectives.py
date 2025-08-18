import numpy as np
import librosa
from scipy.stats import pearsonr
from synthesis import sine_wave

# --- Feature Extraction ---
def compute_mfcc(signal, sample_rate, n_mfcc=20):
    """Computes the mean of MFCCs for a given signal."""
    mfccs = librosa.feature.mfcc(y=signal.astype(np.float32), sr=sample_rate, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1)

def compute_fft(signal, sample_rate, zero_pad_to_next_pow2: bool = True, window: str = 'hann'):
    """
    Computes the positive-frequency FFT magnitude and corresponding frequencies using rFFT.
    Applies a Hann window by default and can zero-pad to the next power-of-two for stability.
    """
    n = len(signal)
    if window == 'hann':
        win = np.hanning(n)
        windowed = signal * win
    else:
        windowed = signal

    if zero_pad_to_next_pow2:
        n_pad = 1 << (n - 1).bit_length()
        if n_pad > n:
            padded = np.zeros(n_pad, dtype=windowed.dtype)
            padded[:n] = windowed
        else:
            padded = windowed
    else:
        padded = windowed

    fft_result = np.fft.rfft(padded)
    fft_freqs = np.fft.rfftfreq(len(padded), 1 / sample_rate)
    fft_magnitude = np.abs(fft_result)
    return fft_magnitude, fft_freqs

# --- Distance Metrics ---
# These now take pre-computed features as input
def mfcc_distance(gen_mfcc, target_mfcc):
    return np.linalg.norm(gen_mfcc - target_mfcc)

def cosine_similarity_dist(gen_spec, target_spec, epsilon: float = 1e-12):
    """
    Cosine distance with safety for zero vectors to avoid warnings.
    Returns 1 - cosine_similarity.
    """
    uu = float(np.dot(gen_spec, gen_spec))
    vv = float(np.dot(target_spec, target_spec))
    if uu <= epsilon or vv <= epsilon:
        return 1.0
    uv = float(np.dot(gen_spec, target_spec))
    return 1.0 - uv / np.sqrt(uu * vv)

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
def prepare_target(
    frequencies,
    amplitudes,
    duration,
    sample_rate,
    objective_type,
    target_spectrum_mode: str = 'sparse',
    gaussian_sigma_hz: float = 30.0,
    fft_zero_pad: bool = True,
    fft_window: str = 'hann',
):
    """
    Pre-computes the target data (MFCC or Spectrum) to avoid re-calculation.

    For spectrum objectives, supports different target representations:
    - 'sparse': impulses at target partial frequencies with normalized amplitudes
    - 'full': rFFT magnitude of the rendered (additive) target signal
    - 'peaks_windowed': sum of Gaussians centered at target partials, weighted by amplitudes
    """
    print("Generating and analyzing target signal...")

    feature_type = METRIC_TYPE.get(objective_type)
    if feature_type is None:
        raise ValueError(f"Unknown objective type: {objective_type}. Not found in METRIC_TYPE dictionary.")

    # Render target only if needed (MFCC or 'full' spectrum)
    target_signal = None
    if feature_type == 'mfcc' or (feature_type == 'spectrum' and target_spectrum_mode == 'full'):
        target_signal = np.zeros(int(sample_rate * duration))
        for freq, amp in zip(frequencies, amplitudes):
            target_signal += sine_wave(freq, amp, duration, sample_rate)
        max_val = np.max(np.abs(target_signal))
        if max_val > 0:
            target_signal /= max_val

    if feature_type == 'mfcc':
        return compute_mfcc(target_signal, sample_rate)

    # Spectrum-based targets
    amplitudes_normalized = amplitudes / np.max(amplitudes)

    # Establish frequency axis length using rFFT settings
    n_base = int(sample_rate * duration)
    if fft_zero_pad:
        n_pad = 1 << (n_base - 1).bit_length()
    else:
        n_pad = n_base
    _, fft_freqs = compute_fft(np.zeros(n_base), sample_rate, zero_pad_to_next_pow2=fft_zero_pad, window=fft_window)

    if target_spectrum_mode == 'sparse':
        target_vec = np.zeros_like(fft_freqs, dtype=np.float64)
        for freq, amp in zip(frequencies, amplitudes_normalized):
            idx = np.argmin(np.abs(fft_freqs - freq))
            target_vec[idx] = amp
        # Normalize to [0,1]
        if np.max(target_vec) > 0:
            target_vec = target_vec / np.max(target_vec)
        return target_vec

    if target_spectrum_mode == 'full':
        target_magnitude, _ = compute_fft(target_signal, sample_rate, zero_pad_to_next_pow2=fft_zero_pad, window=fft_window)
        # Normalize to [0,1]
        max_val = np.max(target_magnitude)
        if max_val > 0:
            target_magnitude = target_magnitude / max_val
        return target_magnitude

    if target_spectrum_mode == 'peaks_windowed':
        target_vec = np.zeros_like(fft_freqs, dtype=np.float64)
        sigma = max(gaussian_sigma_hz, 1e-6)
        for freq, amp in zip(frequencies, amplitudes_normalized):
            # Gaussian over frequency axis
            gauss = np.exp(-0.5 * ((fft_freqs - freq) / sigma) ** 2)
            target_vec += amp * gauss
        # Normalize to [0,1]
        max_val = np.max(target_vec)
        if max_val > 0:
            target_vec = target_vec / max_val
        return target_vec

    raise ValueError(f"Unknown target_spectrum_mode: {target_spectrum_mode}")