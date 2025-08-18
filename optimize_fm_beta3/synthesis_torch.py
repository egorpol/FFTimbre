# synthesis_torch.py
import torch

def synthesize_fm_chain_torch(params: torch.Tensor, duration: float, sample_rate: int):
    """
    Synthesizes a signal using a chain of N FM oscillators using PyTorch.
    The chain is built in reverse: Osc N -> ... -> Osc 1 (Carrier).
    This entire function is differentiable.
    """
    n_oscillators = len(params) // 2
    frequencies = params[0::2]
    amplitudes = params[1::2]
    
    # Create the time vector on the correct device
    time_vector = torch.linspace(0, duration, int(sample_rate * duration), device=params.device)
    
    modulator_signal = torch.zeros_like(time_vector)
    
    # Iterate backwards from the last modulator down to the carrier
    for i in reversed(range(n_oscillators)):
        freq = frequencies[i]
        amp = amplitudes[i]
        # Use torch.sin instead of np.sin
        modulator_signal = amp * torch.sin(2 * torch.pi * freq * time_vector + modulator_signal)
        
    final_signal = modulator_signal
    
    # Normalize the final signal to [-1, 1]
    max_val = torch.max(torch.abs(final_signal))
    if max_val > 0:
        final_signal = final_signal / max_val
        
    return final_signal