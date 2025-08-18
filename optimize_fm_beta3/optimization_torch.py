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
    learning_rate: float = 1e-3,
    fft_zero_pad: bool = True,
    fft_window: str = 'hann',
    loss_mode: str = 'metric', # 'metric' | 'composite'
    w_sc: float = 1.0,
    w_logmag: float = 0.5,
    w_cosine: float = 0.2,
    log_mag_scale: float = 1.0,
    grad_clip_norm: float = 1.0,
    use_adamw: bool = False,
):
    """
    Performs gradient-based optimization using the Adam optimizer in PyTorch.
    """
    # --- START OF FIX: Device Agnostic Code ---
    # 1. Determine the device to use (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Adam optimization on device: {device}")

    # 2. Move all necessary data and parameters to the determined device
    #    Use unconstrained parameters and map to bounds with a sigmoid to avoid clamp-induced plateaus
    init_params = torch.tensor(initial_params, dtype=torch.float32, device=device)
    target_data = target_data.to(device)
    lower_bounds = torch.tensor([b[0] for b in bounds], dtype=torch.float32, device=device)
    upper_bounds = torch.tensor([b[1] for b in bounds], dtype=torch.float32, device=device)
    # Inverse-sigmoid (logit) to initialize raw params close to initial bounded params
    eps = 1e-6
    span = upper_bounds - lower_bounds
    init_norm = (init_params - lower_bounds) / torch.clamp(span, min=eps)
    init_norm = torch.clamp(init_norm, eps, 1 - eps)
    init_raw = torch.log(init_norm / (1 - init_norm))
    raw_params = torch.nn.Parameter(init_raw)
    # --- END OF FIX ---
    
    OptimizerCls = torch.optim.AdamW if use_adamw else torch.optim.Adam
    optimizer = OptimizerCls([raw_params], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)
    
    metric_func = TORCH_METRIC_FUNCTIONS.get(objective_type)
    feature_type = TORCH_METRIC_FEATURE_TYPE.get(objective_type)
    if metric_func is None or feature_type is None:
        raise NotImplementedError(f"Objective type '{objective_type}' is not implemented for PyTorch.")

    error_history = []
    pbar = tqdm(range(max_iters), desc="Optimizing with Adam")

    for i in pbar:
        optimizer.zero_grad()
        # Map raw -> bounded once per step
        bounded_params = lower_bounds + span * torch.sigmoid(raw_params)
        # All tensors generated from bounded_params will now be on the correct device
        generated_signal = synthesize_fm_chain_torch(bounded_params, duration, sample_rate)

        if feature_type == 'spectrum':
            # Apply window then optional zero-padding, then rFFT for positive-frequency spectrum
            n = generated_signal.shape[0]
            if fft_window == 'hann':
                hann = torch.hann_window(n, periodic=True, device=generated_signal.device)
                windowed = generated_signal * hann
            else:
                windowed = generated_signal

            if fft_zero_pad:
                n_pad = 1 << (int(n.item()) - 1).bit_length() if isinstance(n, torch.Tensor) else 1 << (n - 1).bit_length()
                if n_pad > n:
                    padded = torch.zeros(n_pad, device=generated_signal.device, dtype=generated_signal.dtype)
                    padded[:n] = windowed
                else:
                    padded = windowed
            else:
                padded = windowed

            gen_fft = torch.fft.rfft(padded)
            gen_feature = torch.abs(gen_fft)
            max_val = torch.max(gen_feature)
            if max_val > 0: gen_feature = gen_feature / max_val
        elif feature_type == 'mfcc':
            gen_feature = compute_mfcc_torch(generated_signal, sample_rate)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        # Loss calculation now happens entirely on the target device
        if feature_type == 'spectrum' and loss_mode == 'composite':
            # Composite loss: spectral convergence + log-magnitude L1 + cosine distance
            # Ensure shapes match
            if target_data.shape != gen_feature.shape:
                # Interpolate target to match generated length if needed (rare if configs are consistent)
                # Use linear 1D interpolation in frequency domain
                target = target_data
                gen_len = gen_feature.shape[0]
                tgt_len = target.shape[0]
                if tgt_len != gen_len:
                    # Resample via FFT-based padding/cropping is overkill; simple linear interp on index
                    idx_src = torch.linspace(0, 1, steps=tgt_len, device=device)
                    idx_dst = torch.linspace(0, 1, steps=gen_len, device=device)
                    target = torch.interp(idx_dst, idx_src, target)
            else:
                target = target_data

            diff = gen_feature - target
            denom = torch.linalg.norm(target) + 1e-8
            spectral_convergence = torch.linalg.norm(diff) / denom

            log_gen = torch.log1p(log_mag_scale * gen_feature)
            log_tgt = torch.log1p(log_mag_scale * target)
            l1_log_mag = torch.mean(torch.abs(log_gen - log_tgt))

            cosine_dist = 1.0 - torch.nn.functional.cosine_similarity(gen_feature, target, dim=0, eps=1e-8)

            loss = w_sc * spectral_convergence + w_logmag * l1_log_mag + w_cosine * cosine_dist
        else:
            loss = metric_func(gen_feature, target_data)

        loss.backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_([raw_params], max_norm=grad_clip_norm)
        optimizer.step()
        scheduler.step()
        
        loss_item = loss.item()
        error_history.append(loss_item)
        pbar.set_postfix_str(f"Loss: {loss_item:.6f}")

    # Detach and move back to CPU for NumPy conversion
    final_params = (lower_bounds + span * torch.sigmoid(raw_params)).cpu().detach().numpy()
    result = SimpleNamespace(x=final_params, fun=error_history[-1])
    return result, error_history