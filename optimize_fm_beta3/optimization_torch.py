# optimization_torch.py
import torch
from tqdm.auto import tqdm
from types import SimpleNamespace

from synthesis_torch import synthesize_fm_chain_torch
from objectives_torch import TORCH_METRIC_FUNCTIONS, TORCH_METRIC_FEATURE_TYPE, compute_mfcc_torch

def run_adam_optimization(
    objective_type: str,
    target_data: torch.Tensor,
    initial_params: list,
    bounds: list,
    duration: float,
    sample_rate: int,
    max_iters: int,
    learning_rate: float = 1e-2
):
    """
    Performs gradient-based optimization using the Adam optimizer in PyTorch.
    """
    # --- START OF FIX: Device Agnostic Code ---
    # 1. Determine the device to use (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Adam optimization on device: {device}")

    # 2. Move all necessary data and parameters to the determined device
    params_to_optimize = torch.nn.Parameter(torch.tensor(initial_params, dtype=torch.float32, device=device))
    target_data = target_data.to(device)
    lower_bounds = torch.tensor([b[0] for b in bounds], dtype=torch.float32, device=device)
    upper_bounds = torch.tensor([b[1] for b in bounds], dtype=torch.float32, device=device)
    # --- END OF FIX ---
    
    optimizer = torch.optim.Adam([params_to_optimize], lr=learning_rate)
    
    metric_func = TORCH_METRIC_FUNCTIONS.get(objective_type)
    feature_type = TORCH_METRIC_FEATURE_TYPE.get(objective_type)
    if metric_func is None or feature_type is None:
        raise NotImplementedError(f"Objective type '{objective_type}' is not implemented for PyTorch.")

    error_history = []
    pbar = tqdm(range(max_iters), desc="Optimizing with Adam")

    for i in pbar:
        optimizer.zero_grad()
        # All tensors generated from params_to_optimize will now be on the correct device
        generated_signal = synthesize_fm_chain_torch(params_to_optimize, duration, sample_rate)

        if feature_type == 'spectrum':
            gen_fft = torch.fft.fft(generated_signal)
            gen_feature = torch.abs(gen_fft)
            max_val = torch.max(gen_feature)
            if max_val > 0: gen_feature = gen_feature / max_val
        elif feature_type == 'mfcc':
            gen_feature = compute_mfcc_torch(generated_signal, sample_rate)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        # Loss calculation now happens entirely on the target device
        loss = metric_func(gen_feature, target_data)

        loss.backward()
        optimizer.step()
        
        # The clamp operation is now also on the correct device
        with torch.no_grad():
            params_to_optimize.data.clamp_(min=lower_bounds, max=upper_bounds)
        
        loss_item = loss.item()
        error_history.append(loss_item)
        pbar.set_postfix_str(f"Loss: {loss_item:.6f}")

    # Detach and move back to CPU for NumPy conversion
    final_params = params_to_optimize.cpu().detach().numpy()
    result = SimpleNamespace(x=final_params, fun=error_history[-1])
    return result, error_history