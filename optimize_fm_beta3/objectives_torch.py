# objectives_torch.py
import torch
import torch.nn.functional as F
import torchaudio.transforms as T

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
}

# We also need to map objective types to the feature they require
TORCH_METRIC_FEATURE_TYPE = {
    'mfcc_distance': 'mfcc',
    'euclidean_distance': 'spectrum',
    'cosine_similarity': 'spectrum',
    'pearson_correlation_coefficient': 'spectrum',
    'itakura_saito': 'spectrum',
}