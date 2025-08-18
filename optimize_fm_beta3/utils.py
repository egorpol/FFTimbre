# utils.py
import pandas as pd
import numpy as np
import json
import os

def load_data(file_path):
    """Loads frequency and amplitude data from a TSV file."""
    # ... (this function remains the same)
    df = pd.read_csv(file_path, sep='\t')
    frequencies = df['Frequency (Hz)'].values
    amplitudes = df['Amplitude'].values
    amplitudes /= np.max(amplitudes)
    return frequencies, amplitudes

# --- NEW FUNCTIONS FOR PRESET MANAGEMENT ---

def save_preset(filepath: str, params: np.ndarray, metadata: dict):
    """
    Saves FM synthesizer parameters and metadata to a JSON file.

    Args:
        filepath (str): The path to save the JSON file (e.g., 'presets/my_sound.json').
        params (np.ndarray): The array of optimal parameters.
        metadata (dict): A dictionary containing context about the optimization.
    """
    # Ensure the directory exists
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Combine params and metadata into a single dictionary
    # Convert numpy array to a list for JSON serialization
    data_to_save = {
        'metadata': metadata,
        'params': params.tolist() 
    }

    # Write the data to the JSON file
    with open(filepath, 'w') as f:
        json.dump(data_to_save, f, indent=4)
    
    print(f"Preset saved successfully to: {filepath}")

def load_preset(filepath: str) -> tuple[np.ndarray, dict]:
    """
    Loads FM synthesizer parameters and metadata from a JSON file.

    Args:
        filepath (str): The path to the JSON preset file.

    Returns:
        A tuple containing:
        - np.ndarray: The synthesizer parameters.
        - dict: The metadata.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Convert the list of parameters back to a NumPy array
    params = np.array(data['params'])
    metadata = data['metadata']
    
    print(f"Preset loaded from: {filepath}")
    return params, metadata