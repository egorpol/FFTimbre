import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from datetime import datetime

def save_wave_file(
    signal: np.ndarray,
    source_sr: int,
    target_sr: int = 44100,
    bit_depth: int = 16,
    filename: str = None,
    output_dir: str = "rendered_audio"
):
    """
    Resamples, normalizes, and saves an audio signal to a WAV file.

    Args:
        signal (np.ndarray): The input audio signal as a NumPy array.
        source_sr (int): The sample rate of the input signal.
        target_sr (int, optional): The desired sample rate for the output file. Defaults to 44100.
        bit_depth (int, optional): The desired bit depth (16 or 24). Defaults to 16.
        filename (str, optional): The desired filename. If None, a default is generated. Defaults to None.
        output_dir (str, optional): The directory to save the file in. Defaults to "rendered_audio".

    Returns:
        Path: The path to the saved wave file.
    """
    if signal.ndim > 1:
        # soundfile expects shape (n_samples, n_channels)
        # If signal is (n_channels, n_samples), transpose it.
        if signal.shape[0] < signal.shape[1]:
             signal = signal.T
        
    # --- Resampling ---
    if source_sr != target_sr:
        print(f"Resampling signal from {source_sr} Hz to {target_sr} Hz...")
        # Use kaiser_fast for a good speed/quality tradeoff
        signal_resampled = librosa.resample(signal.astype(np.float32), orig_sr=source_sr, target_sr=target_sr, res_type='kaiser_fast')
    else:
        signal_resampled = signal

    # --- Normalization ---
    # Normalize to [-1.0, 1.0] to prevent clipping and for correct integer conversion
    max_val = np.max(np.abs(signal_resampled))
    if max_val > 0:
        signal_normalized = signal_resampled / max_val
    else:
        signal_normalized = signal_resampled # It's all zeros

    # --- File Naming and Path Handling ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True) # Ensure the directory exists

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_audio_{target_sr}Hz_{bit_depth}bit_{timestamp}.wav"

    file_path = output_path / filename

    # --- Writing to File ---
    # soundfile handles the float-to-integer conversion automatically via the subtype
    subtype = f'PCM_{bit_depth}'
    try:
        sf.write(file_path, signal_normalized, target_sr, subtype=subtype)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] File saved successfully to: {file_path}")
    except Exception as e:
        print(f"Error saving file: {e}")
        raise

    return file_path