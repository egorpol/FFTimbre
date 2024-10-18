import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

def mean_squared_error(target_spectrum, optimized_spectrum):
    """Calculate the Mean Squared Error (MSE) between two spectra."""
    return np.mean((target_spectrum - optimized_spectrum) ** 2)

def root_mean_squared_error(target_spectrum, optimized_spectrum):
    """Calculate the Root Mean Squared Error (RMSE) between two spectra."""
    return np.sqrt(mean_squared_error(target_spectrum, optimized_spectrum))

def mean_absolute_error(target_spectrum, optimized_spectrum):
    """Calculate the Mean Absolute Error (MAE) between two spectra."""
    return np.mean(np.abs(target_spectrum - optimized_spectrum))

def cosine_similarity(target_spectrum, optimized_spectrum):
    """Calculate the Cosine Similarity between two spectra."""
    return 1 - cosine(target_spectrum, optimized_spectrum)

def pearson_correlation(target_spectrum, optimized_spectrum):
    """Calculate the Pearson Correlation Coefficient between two spectra."""
    # Check if either spectrum is constant
    if np.all(target_spectrum == target_spectrum[0]) or np.all(optimized_spectrum == optimized_spectrum[0]):
        return np.nan  # Return NaN if correlation is not defined
    return pearsonr(target_spectrum, optimized_spectrum)[0]

def spectral_convergence(target_spectrum, optimized_spectrum):
    """Calculate the Spectral Convergence between two spectra."""
    return np.linalg.norm(optimized_spectrum - target_spectrum) / np.linalg.norm(target_spectrum)

def evaluate_fit(target_freqs, target_amps, optimized_freqs, optimized_amps):
    """Evaluate how well the optimized spectrum fits the target spectrum using various metrics."""
    # Interpolate the optimized spectrum to align with the target frequencies
    optimized_amps_interp = np.interp(target_freqs, optimized_freqs, optimized_amps)
    
    mse = mean_squared_error(target_amps, optimized_amps_interp)
    rmse = root_mean_squared_error(target_amps, optimized_amps_interp)
    mae = mean_absolute_error(target_amps, optimized_amps_interp)
    cosine_sim = cosine_similarity(target_amps, optimized_amps_interp)
    pearson_corr = pearson_correlation(target_amps, optimized_amps_interp)
    spectral_conv = spectral_convergence(target_amps, optimized_amps_interp)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Cosine Similarity: {cosine_sim}")
    print(f"Pearson Correlation Coefficient: {pearson_corr}")
    print(f"Spectral Convergence: {spectral_conv}")
