from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence

import numpy as np

# Local imports from fm_synth_opt
from .fm_synth_opt import (
    FMObjective,
    run_de_optimization,
    run_dual_annealing,
    run_basinhopping,
    synth_chain,
    save_wav,
    save_and_display_final_values,
    make_fft,
)


MethodName = Literal["de", "da", "bh"]
MetricName = FMObjective.__annotations__.get("metric", str)  # keep in sync


@dataclass
class BatchJob:
    """Single batch job specification.

    - method: one of 'de' (differential evolution), 'da' (dual annealing), 'bh' (basin hopping)
    - metric: one of FMObjective metrics ('pearson', 'mfcc', 'itakura_saito', ...)
    - kwargs: optimizer-specific keyword args (e.g., maxiter, popsize, workers, etc.)
    """

    method: MethodName
    metric: str
    kwargs: dict | None = None


def _run_one(
    target_freqs: np.ndarray,
    target_amps: np.ndarray,
    *,
    target_name: Optional[str] = None,
    method: MethodName,
    metric: str,
    sr: int = 44100,
    duration: float = 1.0,
    fft_pad: int = 2,
    fade_in_ms: float = 10.0,
    fade_out_ms: float = 10.0,
    bounds: Optional[Sequence[tuple[float, float]]] = None,
    seed: Optional[int] = 42,
    optimizer_kwargs: Optional[dict] = None,
    save_plots: bool = True,
    plots_dir: str = "rendered_plots",
):
    """Run a single metric/method optimization and save outputs.

    Returns a dict with result info including 'params', 'best', 'wav_path', 'tsv_path'.
    """
    optimizer_kwargs = dict(optimizer_kwargs or {})

    # Build objective
    obj = FMObjective(
        target_freqs=target_freqs,
        target_amps=target_amps,
        metric=metric,
        duration=duration,
        sr=sr,
        fft_pad=fft_pad,
        target_kernel="gaussian",
        target_bw_hz=2.0,
        seed=seed,
    )
    local_bounds = list(bounds) if bounds is not None else obj.default_bounds(
        freq_lo=5.0, freq_hi=5000.0, amp_lo=0.0, amp_hi=10.0
    )

    # Filter kwargs per method to avoid passing unsupported ones
    allowed_keys = {
        "de": {"strategy", "maxiter", "popsize", "tol", "mutation", "recombination", "workers", "seed", "tqdm_desc"},
        "da": {"maxiter", "initial_temp", "seed", "tqdm_desc"},
        "bh": {"maxiter", "stepsize", "seed", "x0", "tqdm_desc"},
    }
    if method not in allowed_keys:
        raise ValueError(f"Unknown method: {method}")
    # keep only supported keys for the chosen method
    optimizer_kwargs = {k: v for k, v in optimizer_kwargs.items() if k in allowed_keys[method]}

    # Dispatch by method
    if method == "de":
        # Provide a friendly default description in tqdm
        optimizer_kwargs.setdefault(
            "tqdm_desc", f"FM 4-osc DE → {str(metric).title()}"
        )
        result, history = run_de_optimization(obj, local_bounds, seed=seed, **optimizer_kwargs)
    elif method == "da":
        optimizer_kwargs.setdefault(
            "tqdm_desc", f"FM 4-osc DA → {str(metric).title()}"
        )
        result, history = run_dual_annealing(obj, local_bounds, seed=seed, **optimizer_kwargs)
    elif method == "bh":
        optimizer_kwargs.setdefault(
            "tqdm_desc", f"FM 4-osc BH → {str(metric).title()}"
        )
        result, history = run_basinhopping(obj, local_bounds, seed=seed, **optimizer_kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    params = np.asarray(result.x, dtype=float)
    best = float(result.fun)

    # Build a shared suffix identical to save_wav(add_info=True, add_time=True)
    # to keep filenames consistent across audio and plots.
    from datetime import datetime
    import os
    info_parts = []
    if method:
        info_parts.append(str(method).strip().lower())
    if metric:
        info_parts.append(str(metric).strip().lower())
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = "_".join([p for p in ("_".join(info_parts), timestamp) if p]) if info_parts else timestamp

    # Synthesize and save audio with a pre-suffixed filename for exact match
    _, y = synth_chain(params, duration=duration, sr=sr)
    # Build base prefix including target name if provided
    safe_target = str(target_name).strip() if target_name else None
    base_prefix = f"optimized_output_fm{('_' + safe_target) if safe_target else ''}"
    base_audio = f"rendered_audio/{base_prefix}"
    wav_path = save_wav(
        f"{base_audio}_{suffix}.wav",
        y,
        sr=sr,
        fade_in_ms=fade_in_ms,
        fade_out_ms=fade_out_ms,
        add_info=False,
        add_time=False,
    )

    # Save TSV of final oscillator values
    tsv_prefix = f"final_values_fm{('_' + safe_target) if safe_target else ''}"
    tsv_path = f"tsv/{tsv_prefix}_{suffix}.tsv"
    _ = save_and_display_final_values(params, tsv_path)

    # Optionally save diagnostic plots using the same suffix
    plot_paths = {}
    if save_plots:
        import matplotlib.pyplot as plt

        os.makedirs(plots_dir, exist_ok=True)

        # Time plot
        plt.figure(figsize=(10, 4))
        n_samples = min(1000, y.shape[0])
        plt.plot(y[:n_samples])
        plt.title("Optimized Combined Signal (time domain)")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        p_time = os.path.join(plots_dir, f"{base_prefix}_{suffix}_time.png")
        plt.savefig(p_time, dpi=150)
        plt.close()
        plot_paths["time"] = p_time

        # Spectrum plot (normalized), overlay target partials
        freqs, mag = make_fft(y, sr=sr, fft_pad=fft_pad)
        mag = mag / (np.max(mag) + 1e-12)
        plt.figure(figsize=(10, 4))
        plt.plot(freqs, mag, label="Optimized spectrum")
        if target_freqs is not None and target_amps is not None:
            t_amps = target_amps / (np.max(target_amps) + 1e-12)
            plt.scatter(target_freqs, t_amps, label="Target spectrum", color='red', zorder=5)
        plt.xlim(0, 8000)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (norm)")
        plt.title("Frequency Spectrum of Optimized Signal")
        plt.legend()
        plt.tight_layout()
        p_spec = os.path.join(plots_dir, f"{base_prefix}_{suffix}_spectrum.png")
        plt.savefig(p_spec, dpi=150)
        plt.close()
        plot_paths["spectrum"] = p_spec

        # Error history plot
        plt.figure(figsize=(10, 4))
        plt.plot(history)
        plt.title("Error progression")
        plt.xlabel("Iteration/Generation")
        plt.ylabel("Objective value")
        plt.grid(True)
        plt.tight_layout()
        p_err = os.path.join(plots_dir, f"{base_prefix}_{suffix}_error.png")
        plt.savefig(p_err, dpi=150)
        plt.close()
        plot_paths["error"] = p_err

    return {
        "method": method,
        "metric": metric,
        "best": best,
        "params": params,
        "history": history,
        "wav_path": wav_path,
        "tsv_path": tsv_path,
        "plots": plot_paths,
    }


def run_batch_jobs(
    target_freqs: np.ndarray,
    target_amps: np.ndarray,
    jobs: Iterable[BatchJob | dict],
    *,
    target_name: Optional[str] = None,
    sr: int = 44100,
    duration: float = 1.0,
    fft_pad: int = 2,
    fade_in_ms: float = 10.0,
    fade_out_ms: float = 10.0,
    bounds: Optional[Sequence[tuple[float, float]]] = None,
    seed: Optional[int] = 42,
):
    """Run a list of metric/optimizer jobs sequentially and collect results.

    Example job specs:
        BatchJob(method='de', metric='pearson', kwargs=dict(maxiter=300, workers=-1))
        {"method": "da", "metric": "mfcc", "kwargs": {"maxiter": 400}}

    Returns a list of dict rows suitable for constructing a pandas DataFrame.
    """
    rows = []
    for j in jobs:
        if isinstance(j, dict):
            method = j.get("method")
            metric = j.get("metric")
            kwargs = j.get("kwargs") or {}
        else:  # BatchJob dataclass
            method = j.method
            metric = j.metric
            kwargs = dict(j.kwargs or {})

        info = _run_one(
            target_freqs,
            target_amps,
            target_name=target_name,
            method=method,  # type: ignore[arg-type]
            metric=str(metric),
            sr=sr,
            duration=duration,
            fft_pad=fft_pad,
            fade_in_ms=fade_in_ms,
            fade_out_ms=fade_out_ms,
            bounds=bounds,
            seed=seed,
            optimizer_kwargs=kwargs,
        )
        rows.append(info)

    return rows
