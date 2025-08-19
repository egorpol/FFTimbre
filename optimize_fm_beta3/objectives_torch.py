# objectives_torch.py
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from typing import Optional

# --- Feature Extraction (PyTorch native) ---

def compute_mfcc_torch(signal: torch.Tensor, sample_rate: int, n_mfcc: int = 20) -> torch.Tensor:
    """Computes MFCCs for a given signal using torchaudio."""
    # torchaudio expects the input to be (..., time)
    # The MFCC transform needs to be initialized first
    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
    )
    # The transform needs to be on the same device as the signal
    mfcc_transform = mfcc_transform.to(signal.device)
    mfccs = mfcc_transform(signal)
    # We take the mean across the time dimension to get a single feature vector
    return torch.mean(mfccs, dim=1)


# --- Distance Metrics (PyTorch native) ---

def mfcc_distance_torch(gen_mfcc: torch.Tensor, target_mfcc: torch.Tensor) -> torch.Tensor:
    """Computes Euclidean distance between two MFCC feature vectors."""
    return torch.linalg.norm(gen_mfcc - target_mfcc)

def euclidean_distance_torch(gen_spec: torch.Tensor, target_spec: torch.Tensor) -> torch.Tensor:
    """Computes Euclidean (L2) distance between two spectra."""
    return torch.linalg.norm(gen_spec - target_spec)

def cosine_similarity_dist_torch(gen_spec: torch.Tensor, target_spec: torch.Tensor) -> torch.Tensor:
    """Computes Cosine Distance (1 - Cosine Similarity) between two spectra."""
    # F.cosine_similarity expects at least 1D, so our inputs are fine.
    # We add an epsilon to the denominator for numerical stability, though often not needed.
    return 1.0 - F.cosine_similarity(gen_spec, target_spec, dim=0, eps=1e-8)

def pearson_correlation_dist_torch(gen_spec: torch.Tensor, target_spec: torch.Tensor) -> torch.Tensor:
    """
    Computes Pearson Correlation Distance (1 - Pearson Corr Coeff) between two spectra.
    Pearson(x, y) = Cov(x, y) / (std(x) * std(y))
    """
    # Center the vectors by subtracting their means
    gen_centered = gen_spec - torch.mean(gen_spec)
    target_centered = target_spec - torch.mean(target_spec)
    
    # Compute the covariance
    covariance = torch.sum(gen_centered * target_centered)
    
    # Compute the standard deviations and their product
    gen_std = torch.sqrt(torch.sum(gen_centered**2))
    target_std = torch.sqrt(torch.sum(target_centered**2))
    std_product = gen_std * target_std
    
    # Calculate the Pearson correlation coefficient
    # Add a small epsilon for numerical stability in case of zero standard deviation
    correlation = covariance / (std_product + 1e-8)
    
    # We want to minimize the distance, so we return 1 - correlation
    return 1.0 - correlation

def itakura_saito_dist_torch(gen_spec: torch.Tensor, target_spec: torch.Tensor) -> torch.Tensor:
    """
    Computes Itakura-Saito divergence between two spectra.
    Typically used with power spectra, but applied here to magnitude.
    """
    epsilon = 1e-10  # Small constant to avoid division by zero and log(0)
    
    # Ensure spectra are positive (clamp at epsilon)
    gen_spec_clamped = torch.clamp(gen_spec, min=epsilon)
    target_spec_clamped = torch.clamp(target_spec, min=epsilon)
    
    # IS divergence formula: sum( target/gen - log(target/gen) - 1 )
    ratio = target_spec_clamped / gen_spec_clamped
    log_ratio = torch.log(ratio)
    
    is_distance = torch.sum(ratio - log_ratio - 1)
    
    return is_distance

def log_spectral_distance_torch(gen_spec: torch.Tensor, target_spec: torch.Tensor, epsilon: float = 1e-12, use_db: bool = True) -> torch.Tensor:
    """LSD: RMSE of log-magnitude difference (dB or natural log)."""
    gen = torch.clamp(gen_spec, min=epsilon)
    tgt = torch.clamp(target_spec, min=epsilon)
    if use_db:
        gen_log = 20.0 * torch.log10(gen)
        tgt_log = 20.0 * torch.log10(tgt)
    else:
        gen_log = torch.log(gen)
        tgt_log = torch.log(tgt)
    diff = gen_log - tgt_log
    return torch.sqrt(torch.mean(diff * diff))

def log_spectral_distance_weighted_torch(
    gen_spec: torch.Tensor,
    target_spec: torch.Tensor,
    epsilon: float = 1e-12,
    weight_floor: float = 0.05,
    weight_power: float = 1.0,
) -> torch.Tensor:
    """Weighted LSD with log1p compression and target-derived weights."""
    gen = torch.clamp(gen_spec, min=0.0)
    tgt = torch.clamp(target_spec, min=0.0)
    gen_log = torch.log1p(gen)
    tgt_log = torch.log1p(tgt)
    diff_sq = (gen_log - tgt_log) ** 2
    w = tgt ** weight_power
    w_sum = torch.sum(w)
    w = torch.where(w_sum > 0, w / (w_sum + 1e-12), torch.full_like(w, 1.0 / w.numel()))
    w = (1.0 - weight_floor) * w + weight_floor * (1.0 / w.numel())
    return torch.sqrt(torch.sum(w * diff_sq))

def _to_prob_torch(vec: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    v = torch.clamp(vec, min=0.0)
    s = torch.sum(v)
    return v / (s + epsilon)

def kl_divergence_torch(gen_spec: torch.Tensor, target_spec: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    """D_KL(P||Q) with P=target, Q=gen on normalized spectra."""
    P = _to_prob_torch(target_spec, epsilon)
    Q = _to_prob_torch(gen_spec, epsilon)
    ratio = torch.clamp(P, min=epsilon) / torch.clamp(Q, min=epsilon)
    return torch.sum(P * torch.log(ratio))

def jensen_shannon_torch(gen_spec: torch.Tensor, target_spec: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    P = _to_prob_torch(target_spec, epsilon)
    Q = _to_prob_torch(gen_spec, epsilon)
    M = 0.5 * (P + Q)
    def _kl(a, b):
        return torch.sum(a * (torch.log(torch.clamp(a, min=epsilon)) - torch.log(torch.clamp(b, min=epsilon))))
    return 0.5 * _kl(P, M) + 0.5 * _kl(Q, M)

def hellinger_torch(gen_spec: torch.Tensor, target_spec: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    P = _to_prob_torch(target_spec, epsilon)
    Q = _to_prob_torch(gen_spec, epsilon)
    return torch.norm(torch.sqrt(P) - torch.sqrt(Q)) / (2.0 ** 0.5)

def bhattacharyya_torch(gen_spec: torch.Tensor, target_spec: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    P = _to_prob_torch(target_spec, epsilon)
    Q = _to_prob_torch(gen_spec, epsilon)
    bc = torch.sum(torch.sqrt(P * Q))
    bc = torch.clamp(bc, min=epsilon, max=1.0)
    return -torch.log(bc)

def beta_divergence_torch(gen_spec: torch.Tensor, target_spec: torch.Tensor, beta: float, epsilon: float = 1e-12) -> torch.Tensor:
    X = torch.clamp(target_spec, min=epsilon)
    Y = torch.clamp(gen_spec, min=epsilon)
    if abs(beta - 1.0) < 1e-9:
        return kl_divergence_torch(Y, X, epsilon)  # D_KL(X||Y)
    if abs(beta) < 1e-9:
        return itakura_saito_dist_torch(Y, X)      # D_IS(X||Y)
    term = (X ** beta + (beta - 1.0) * (Y ** beta) - beta * X * (Y ** (beta - 1.0)))
    return torch.sum(term) / (beta * (beta - 1.0))

def log_mse_l2_torch(gen_vec: torch.Tensor, target_vec: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    gen_log = torch.log(torch.clamp(gen_vec, min=epsilon))
    tgt_log = torch.log(torch.clamp(target_vec, min=epsilon))
    diff = gen_log - tgt_log
    return torch.sqrt(torch.mean(diff * diff))

def log_mse_l1_torch(gen_vec: torch.Tensor, target_vec: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    gen_log = torch.log(torch.clamp(gen_vec, min=epsilon))
    tgt_log = torch.log(torch.clamp(target_vec, min=epsilon))
    return torch.mean(torch.abs(gen_log - tgt_log))

def _hz_to_erb_rate_torch(f_hz: torch.Tensor) -> torch.Tensor:
    return 21.4 * torch.log10(1.0 + 4.37e-3 * f_hz)

def build_erb_filterbank_torch(freqs: torch.Tensor, n_bands: int = 40, min_hz: float = 20.0, max_hz: Optional[float] = None) -> torch.Tensor:
    """Triangular ERB-spaced filterbank over frequency axis. Returns (n_bands, n_bins)."""
    device = freqs.device
    dtype = freqs.dtype
    if max_hz is None:
        max_hz = float(freqs[-1].item())
    freqs_erb = _hz_to_erb_rate_torch(freqs)
    lo_e = float(_hz_to_erb_rate_torch(torch.tensor([min_hz], device=device, dtype=dtype))[0].item())
    hi_e = float(_hz_to_erb_rate_torch(torch.tensor([max_hz], device=device, dtype=dtype))[0].item())
    edges_e = torch.linspace(lo_e, hi_e, steps=n_bands + 2, device=device, dtype=dtype)
    centers_e = edges_e[1:-1]
    left_e = edges_e[:-2]
    right_e = edges_e[2:]
    W = torch.zeros((n_bands, freqs.shape[0]), device=device, dtype=dtype)
    for i in range(n_bands):
        up = (freqs_erb - left_e[i]) / max((centers_e[i] - left_e[i]).item(), 1e-12)
        down = (right_e[i] - freqs_erb) / max((right_e[i] - centers_e[i]).item(), 1e-12)
        tri = torch.clamp(torch.minimum(up, down), min=0.0)
        s = torch.sum(tri)
        if s > 0:
            tri = tri / s
        W[i, :] = tri
    return W


def build_target_spectrum_torch(
    frequencies: torch.Tensor,
    amplitudes: torch.Tensor,
    sample_rate: int,
    duration: float,
    mode: str = 'sparse',
    gaussian_sigma_hz: float = 30.0,
    fft_zero_pad: bool = True,
    fft_window: str = 'hann',
):
    """
    Build a target spectrum vector on positive frequencies, similar to NumPy path, but in Torch.
    Useful when doing fully-in-Torch objectives.
    """
    n_base = int(sample_rate * duration)
    if fft_zero_pad:
        n_pad = 1 << (n_base - 1).bit_length()
    else:
        n_pad = n_base
    # Frequency axis for rFFT
    freqs = torch.fft.rfftfreq(n_pad, d=1.0 / sample_rate)

    # Normalize amplitudes to [0,1]
    amp_max = torch.max(amplitudes)
    amps_norm = amplitudes / (amp_max + 1e-12)

    if mode == 'sparse':
        target_vec = torch.zeros_like(freqs)
        # Assign to nearest bin
        for f, a in zip(frequencies, amps_norm):
            idx = torch.argmin(torch.abs(freqs - f))
            target_vec[idx] = torch.maximum(target_vec[idx], a)
        # Normalize
        tmax = torch.max(target_vec)
        if tmax > 0:
            target_vec = target_vec / tmax
        return target_vec

    if mode == 'peaks_windowed':
        target_vec = torch.zeros_like(freqs)
        sigma = max(gaussian_sigma_hz, 1e-6)
        for f, a in zip(frequencies, amps_norm):
            gauss = torch.exp(-0.5 * ((freqs - f) / sigma) ** 2)
            target_vec += a * gauss
        tmax = torch.max(target_vec)
        if tmax > 0:
            target_vec = target_vec / tmax
        return target_vec

    if mode == 'full':
        # Render additive target on-the-fly here would require a synth in torch.
        # For now, recommend using NumPy path for 'full' or provide a torch signal externally.
        raise NotImplementedError("'full' mode for torch target spectrum requires a torch-target signal.")

    raise ValueError(f"Unknown mode: {mode}")


# --- Dictionaries to map objective types to their PyTorch functions ---

TORCH_METRIC_FUNCTIONS = {
    'mfcc_distance': mfcc_distance_torch,
    'euclidean_distance': euclidean_distance_torch,
    'cosine_similarity': cosine_similarity_dist_torch,
    'pearson_correlation_coefficient': pearson_correlation_dist_torch,
    'itakura_saito': itakura_saito_dist_torch,
    'log_spectral_distance': log_spectral_distance_torch,
    'kl_divergence': kl_divergence_torch,
    'jensen_shannon': jensen_shannon_torch,
    'hellinger': hellinger_torch,
    'bhattacharyya': bhattacharyya_torch,
    'log_spectral_distance_weighted': log_spectral_distance_weighted_torch,
    'erb_log_l2': log_mse_l2_torch,
    'erb_log_l1': log_mse_l1_torch,
}

# We also need to map objective types to the feature they require
TORCH_METRIC_FEATURE_TYPE = {
    'mfcc_distance': 'mfcc',
    'euclidean_distance': 'spectrum',
    'cosine_similarity': 'spectrum',
    'pearson_correlation_coefficient': 'spectrum',
    'itakura_saito': 'spectrum',
    'log_spectral_distance': 'spectrum',
    'kl_divergence': 'spectrum',
    'jensen_shannon': 'spectrum',
    'hellinger': 'spectrum',
    'bhattacharyya': 'spectrum',
    'erb_log_l2': 'erb',
    'erb_log_l1': 'erb',
}

def resolve_metric_torch(objective_type: str):
    """Resolve an objective type to (metric_func, feature_type) in torch.
    Supports parameterized 'beta_divergence:beta'."""
    name = objective_type.strip().lower()
    if name.startswith('beta_divergence'):
        beta = 1.0
        if ':' in name:
            try:
                beta = float(name.split(':', 1)[1])
            except Exception:
                beta = 1.0
        def metric(gen_spec, target_spec, _beta=beta):
            return beta_divergence_torch(gen_spec, target_spec, beta=_beta)
        return metric, 'spectrum'
    if name in TORCH_METRIC_FUNCTIONS and name in TORCH_METRIC_FEATURE_TYPE:
        return TORCH_METRIC_FUNCTIONS[name], TORCH_METRIC_FEATURE_TYPE[name]
    raise ValueError(f"Objective type '{objective_type}' is not implemented for PyTorch.")