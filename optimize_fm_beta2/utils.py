# utils.py
import pandas as pd
import numpy as np

def load_data(file_path):
    """Loads frequency and amplitude data from a TSV file."""
    df = pd.read_csv(file_path, sep='\t')
    frequencies = df['Frequency (Hz)'].values
    amplitudes = df['Amplitude'].values
    # Normalize amplitudes to a [0, 1] range
    amplitudes /= np.max(amplitudes)
    return frequencies, amplitudes