# synthesis.py
import numpy as np

def sine_wave(frequency, amplitude, duration, sample_rate):
    """Generates a simple sine wave."""
    time_vector = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * frequency * time_vector)

def synthesize_fm_chain(params, duration, sample_rate):
    """
    Synthesizes a signal using a chain of N FM oscillators.
    The chain is built in reverse: Osc N -> N-1 -> ... -> Osc 2 -> Osc 1 (Carrier)
    Params are expected as [freq_1, amp_1, freq_2, amp_2, ...].
    """
    n_oscillators = len(params) // 2
    # NOTE: The parameters are now conceptually ordered from Carrier to last Modulator
    # i.e., params[0], params[1] are for Oscillator 1 (the Carrier)
    frequencies = params[0::2]
    amplitudes = params[1::2]
    
    time_vector = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    modulator_signal = np.zeros_like(time_vector)
    
    # --- THIS IS THE KEY CHANGE ---
    # Iterate backwards from the last modulator (N) down to the carrier (1)
    for i in reversed(range(n_oscillators)):
        freq = frequencies[i]
        amp = amplitudes[i]
        # The new signal is modulated by the previous one in the chain
        modulator_signal = amp * np.sin(2 * np.pi * freq * time_vector + modulator_signal)
        
    final_signal = modulator_signal
    
    max_val = np.max(np.abs(final_signal))
    if max_val > 0:
        final_signal /= max_val
        
    return final_signal