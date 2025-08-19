import numpy as np
import librosa
from scipy.stats import pearsonr
from synthesis import sine_wave
from typing import Optional

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

# --- Additional Distance Metrics (single-frame spectrum) ---

def log_spectral_distance(gen_spec: np.ndarray, target_spec: np.ndarray, epsilon: float = 1e-12, use_db: bool = False) -> float:
    """
    Log-Spectral Distance (LSD): RMSE of log-magnitude difference.
    If use_db=True, uses 20*log10; otherwise uses natural log on magnitudes.
    """
    gen = np.maximum(gen_spec, epsilon)
    tgt = np.maximum(target_spec, epsilon)
    if use_db:
        gen_log = 20.0 * np.log10(gen)
        tgt_log = 20.0 * np.log10(tgt)
    else:
        gen_log = np.log(gen)
        tgt_log = np.log(tgt)
    diff_sq = (gen_log - tgt_log) ** 2
    return float(np.sqrt(np.mean(diff_sq)))

def log_spectral_distance_weighted(
    gen_spec: np.ndarray,
    target_spec: np.ndarray,
    epsilon: float = 1e-12,
    weight_floor: float = 0.05,
    weight_power: float = 1.0,
) -> float:
    """
    Weighted LSD to avoid collapse with sparse or peaky targets.
    - Uses log1p compression to stabilize low bins and zeros.
    - Weights bins by target magnitude^power with a small uniform floor so background contributes.
    """
    gen = np.maximum(gen_spec, 0.0)
    tgt = np.maximum(target_spec, 0.0)
    gen_log = np.log1p(gen)
    tgt_log = np.log1p(tgt)
    diff_sq = (gen_log - tgt_log) ** 2
    w = tgt ** weight_power
    w_sum = np.sum(w)
    if w_sum > 0:
        w = w / w_sum
    # add uniform floor
    w = (1.0 - weight_floor) * w + weight_floor * (1.0 / len(w))
    return float(np.sqrt(np.sum(w * diff_sq)))

def _to_prob(vec: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    v = np.maximum(vec.astype(np.float64), 0.0)
    s = np.sum(v)
    if s <= epsilon:
        # fallback to uniform to avoid NaNs; does not contribute to meaningful gradient but safe for eval
        return np.full_like(v, 1.0 / len(v))
    return v / s

def kl_divergence(gen_spec: np.ndarray, target_spec: np.ndarray, epsilon: float = 1e-12) -> float:
    """
    KL divergence D_KL(P || Q) where P=target, Q=gen on normalized spectra.
    """
    P = _to_prob(target_spec, epsilon)
    Q = _to_prob(gen_spec, epsilon)
    ratio = np.maximum(P, epsilon) / np.maximum(Q, epsilon)
    return float(np.sum(P * np.log(ratio)))

def jensen_shannon_divergence(gen_spec: np.ndarray, target_spec: np.ndarray, epsilon: float = 1e-12) -> float:
    """
    Jensen–Shannon divergence between two normalized spectra (bounded, symmetric).
    """
    P = _to_prob(target_spec, epsilon)
    Q = _to_prob(gen_spec, epsilon)
    M = 0.5 * (P + Q)
    def _kl(a, b):
        return np.sum(a * (np.log(np.maximum(a, epsilon)) - np.log(np.maximum(b, epsilon))))
    return float(0.5 * _kl(P, M) + 0.5 * _kl(Q, M))

def hellinger_distance(gen_spec: np.ndarray, target_spec: np.ndarray, epsilon: float = 1e-12) -> float:
    """
    Hellinger distance between two normalized spectra.
    H(P,Q) = (1/sqrt(2)) * || sqrt(P) - sqrt(Q) ||_2
    """
    P = _to_prob(target_spec, epsilon)
    Q = _to_prob(gen_spec, epsilon)
    return float(np.linalg.norm(np.sqrt(P) - np.sqrt(Q)) / np.sqrt(2.0))

def bhattacharyya_distance(gen_spec: np.ndarray, target_spec: np.ndarray, epsilon: float = 1e-12) -> float:
    """Bhattacharyya distance: -log( sum_i sqrt(P_i Q_i) )."""
    P = _to_prob(target_spec, epsilon)
    Q = _to_prob(gen_spec, epsilon)
    bc = np.sum(np.sqrt(P * Q))
    bc = np.clip(bc, epsilon, 1.0)
    return float(-np.log(bc))

def beta_divergence(gen_spec: np.ndarray, target_spec: np.ndarray, beta: float, epsilon: float = 1e-12) -> float:
    """
    General β-divergence (on nonnegative spectra). Special cases:
      β=2 -> (1/2)||x-y||^2,  β->1 -> KL,  β->0 -> IS
    Implements the standard definition for β not in {0,1}.
    Note: For KL/IS, prefer kl_divergence/itakura_saito_dist for numerical stability.
    """
    X = np.maximum(target_spec.astype(np.float64), epsilon)
    Y = np.maximum(gen_spec.astype(np.float64), epsilon)
    if abs(beta - 1.0) < 1e-9:
        return kl_divergence(Y, X, epsilon)  # D_KL(X||Y)
    if abs(beta) < 1e-9:
        return itakura_saito_dist(Y, X)      # D_IS(X||Y)
    term = (X ** beta + (beta - 1.0) * (Y ** beta) - beta * X * (Y ** (beta - 1.0)))
    return float(np.sum(term) / (beta * (beta - 1.0)))

# --- ERB-band utilities ---

def _hz_to_erb_rate(f_hz: np.ndarray) -> np.ndarray:
    return 21.4 * np.log10(1.0 + 4.37e-3 * f_hz)

def _erb_rate_to_hz(e: np.ndarray) -> np.ndarray:
    return (10 ** (e / 21.4) - 1.0) / 4.37e-3

def build_erb_filterbank(fft_freqs: np.ndarray, n_bands: int = 40, min_hz: float = 20.0, max_hz: Optional[float] = None) -> np.ndarray:
    """
    Build a simple triangular ERB-spaced filterbank over the provided frequency axis.
    Returns weights of shape (n_bands, n_bins), approximately area-normalized per band.
    """
    if max_hz is None:
        max_hz = float(fft_freqs[-1])
    freqs = np.asarray(fft_freqs, dtype=np.float64)
    freqs_erb = _hz_to_erb_rate(freqs)
    lo_e = float(_hz_to_erb_rate(np.array([min_hz]))[0])
    hi_e = float(_hz_to_erb_rate(np.array([max_hz]))[0])
    edges_e = np.linspace(lo_e, hi_e, num=n_bands + 2)
    centers_e = edges_e[1:-1]
    left_e = edges_e[:-2]
    right_e = edges_e[2:]
    W = np.zeros((n_bands, len(freqs)), dtype=np.float64)
    for i in range(n_bands):
        # triangular in ERB-rate domain
        up = (freqs_erb - left_e[i]) / max(centers_e[i] - left_e[i], 1e-12)
        down = (right_e[i] - freqs_erb) / max(right_e[i] - centers_e[i], 1e-12)
        tri = np.maximum(0.0, np.minimum(up, down))
        # area-normalize per band to sum to 1 over bins with positive weight
        s = np.sum(tri)
        if s > 0:
            tri = tri / s
        W[i, :] = tri
    return W

def log_mse_l2(gen_vec: np.ndarray, target_vec: np.ndarray, epsilon: float = 1e-12) -> float:
    gen_log = np.log(np.maximum(gen_vec, epsilon))
    tgt_log = np.log(np.maximum(target_vec, epsilon))
    return float(np.sqrt(np.mean((gen_log - tgt_log) ** 2)))

def log_mse_l1(gen_vec: np.ndarray, target_vec: np.ndarray, epsilon: float = 1e-12) -> float:
    gen_log = np.log(np.maximum(gen_vec, epsilon))
    tgt_log = np.log(np.maximum(target_vec, epsilon))
    return float(np.mean(np.abs(gen_log - tgt_log)))

# +++ UPDATE THE DICTIONARIES +++
METRIC_FUNCTIONS = {
    'mfcc_distance': mfcc_distance,
    'cosine_similarity': cosine_similarity_dist,
    'euclidean_distance': euclidean_distance,
    'pearson_correlation_coefficient': pearson_correlation_dist,
    'itakura_saito': itakura_saito_dist,
    # New spectrum metrics
    'log_spectral_distance': log_spectral_distance,
    'kl_divergence': kl_divergence,
    'jensen_shannon': jensen_shannon_divergence,
    'hellinger': hellinger_distance,
    'bhattacharyya': bhattacharyya_distance,
    'log_spectral_distance_weighted': log_spectral_distance_weighted,
    # ERB-band metrics operate on ERB-band energies (handled by optimizer to pre-project)
    'erb_log_l2': log_mse_l2,
    'erb_log_l1': log_mse_l1,
}

METRIC_TYPE = {
    'mfcc_distance': 'mfcc',
    'cosine_similarity': 'spectrum',
    'euclidean_distance': 'spectrum',
    'pearson_correlation_coefficient': 'spectrum',
    'itakura_saito': 'spectrum',
    'log_spectral_distance': 'spectrum',
    'kl_divergence': 'spectrum',
    'jensen_shannon': 'spectrum',
    'hellinger': 'spectrum',
    'bhattacharyya': 'spectrum',
    'log_spectral_distance_weighted': 'spectrum',
    'erb_log_l2': 'erb',
    'erb_log_l1': 'erb',
}

def resolve_metric(objective_type: str):
    """
    Resolve an objective name into (metric_func, feature_type).
    Supports parameterized beta-divergence via names like 'beta_divergence:1.2'.
    """
    name = objective_type.strip().lower()
    if name.startswith('beta_divergence'):
        beta = 1.0
        if ':' in name:
            try:
                beta = float(name.split(':', 1)[1])
            except Exception:
                beta = 1.0
        def metric(gen_spec, target_spec, _beta=beta):
            return beta_divergence(gen_spec, target_spec, beta=_beta)
        return metric, 'spectrum'
    if name in METRIC_FUNCTIONS and name in METRIC_TYPE:
        return METRIC_FUNCTIONS[name], METRIC_TYPE[name]
    raise ValueError(f"Unknown objective type: {objective_type}")

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
    erb_n_bands: int = 40,
    erb_min_hz: float = 20.0,
    erb_max_hz: Optional[float] = None,
):
    """
    Pre-computes the target data (MFCC or Spectrum) to avoid re-calculation.

    For spectrum objectives, supports different target representations:
    - 'sparse': impulses at target partial frequencies with normalized amplitudes
    - 'full': rFFT magnitude of the rendered (additive) target signal
    - 'peaks_windowed': sum of Gaussians centered at target partials, weighted by amplitudes
    """
    print("Generating and analyzing target signal...")

    try:
        _, feature_type = resolve_metric(objective_type)
    except Exception:
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

    # Spectrum-based (and ERB) targets
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
        if feature_type == 'spectrum':
            return target_vec
        elif feature_type == 'erb':
            W = build_erb_filterbank(fft_freqs, n_bands=erb_n_bands, min_hz=erb_min_hz, max_hz=erb_max_hz)
            band_energies = W @ target_vec
            m = np.max(band_energies)
            if m > 0:
                band_energies = band_energies / m
            return band_energies

    if target_spectrum_mode == 'full':
        target_magnitude, _ = compute_fft(target_signal, sample_rate, zero_pad_to_next_pow2=fft_zero_pad, window=fft_window)
        # Normalize to [0,1]
        max_val = np.max(target_magnitude)
        if max_val > 0:
            target_magnitude = target_magnitude / max_val
        if feature_type == 'spectrum':
            return target_magnitude
        elif feature_type == 'erb':
            W = build_erb_filterbank(fft_freqs, n_bands=erb_n_bands, min_hz=erb_min_hz, max_hz=erb_max_hz)
            band_energies = W @ target_magnitude
            m = np.max(band_energies)
            if m > 0:
                band_energies = band_energies / m
            return band_energies

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
        if feature_type == 'spectrum':
            return target_vec
        elif feature_type == 'erb':
            W = build_erb_filterbank(fft_freqs, n_bands=erb_n_bands, min_hz=erb_min_hz, max_hz=erb_max_hz)
            band_energies = W @ target_vec
            m = np.max(band_energies)
            if m > 0:
                band_energies = band_energies / m
            return band_energies

    raise ValueError(f"Unknown target_spectrum_mode: {target_spectrum_mode}")