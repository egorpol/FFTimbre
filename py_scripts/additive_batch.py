from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Callable, Literal

import numpy as np
import matplotlib.pyplot as plt

from .additive_synth_opt import (
    AdditiveObjective,
    additive_synth,
    save_and_display_final_values_additive,
    make_fft,
)
from .fm_synth_opt import save_wav
from .waveform_generators import (
    sine_wave,
    square_wave,
    triangle_wave,
    sawtooth_wave,
    noise_wave,
)
from .fm_synth_opt import (
    run_de_optimization,
    run_dual_annealing,
    run_basinhopping,
)


DEFAULT_WAVEFORMS: list[Callable[[float, float, float, int], np.ndarray]] = [
    sine_wave,
    square_wave,
    triangle_wave,
    sawtooth_wave,
    noise_wave,
]


OptimizerMethod = Literal["de", "da", "bh"]


@dataclass
class RunAdditiveConfig:
    # Target
    target_freqs: np.ndarray
    target_amps: np.ndarray
    target_name: str

    # Synthesis
    n_partials: int = 4
    use_waveforms: bool = True
    waveforms: Optional[Sequence[Callable[[float, float, float, int], np.ndarray]]] = tuple(DEFAULT_WAVEFORMS)

    # Audio / analysis settings
    sr: int = 44100
    duration: float = 2.0
    fft_pad: int = 2
    waveform_dtype: str = "float32"  # control synthesis dtype for waveforms/additive
    fade_in_ms: float = 10.0
    fade_out_ms: float = 10.0

    # Optimization
    method: OptimizerMethod = "de"
    metric: str = "pearson"
    optimizer_kwargs: dict = None
    seed: Optional[int] = 42

    # Outputs
    audio_dir: str = "rendered_audio"
    plots_dir: str = "rendered_plots"
    tsv_dir: str = "tsv"


def run_one_additive(cfg: RunAdditiveConfig) -> dict:
    """Run a single additive optimization and save audio/plots/tsv.

    Returns a dict with: method, metric, best, params, history, wav_path, tsv_path, plots.
    """
    optimizer_kwargs = dict(cfg.optimizer_kwargs or {})
    # dtype selection (used for both objective and synthesis)
    np_dtype = np.float64 if str(cfg.waveform_dtype).lower() == "float64" else np.float32

    obj = AdditiveObjective(
        target_freqs=cfg.target_freqs,
        target_amps=cfg.target_amps,
        metric=cfg.metric,
        n_partials=cfg.n_partials,
        duration=cfg.duration,
        sr=cfg.sr,
        fft_pad=cfg.fft_pad,
        target_kernel="gaussian",
        target_bw_hz=2.0,
        seed=cfg.seed,
        waveforms=(cfg.waveforms if cfg.use_waveforms else None),
        dtype=np_dtype,
    )
    bounds = obj.default_bounds(freq_lo=5.0, freq_hi=5000.0, amp_lo=0.0, amp_hi=1.0)

    allowed = {
        "de": {"strategy", "maxiter", "popsize", "tol", "mutation", "recombination", "workers", "seed", "tqdm_desc"},
        "da": {"maxiter", "initial_temp", "seed", "tqdm_desc"},
        "bh": {"maxiter", "stepsize", "seed", "x0", "tqdm_desc"},
    }
    if cfg.method not in allowed:
        raise ValueError(f"Unknown method: {cfg.method}")
    optimizer_kwargs = {k: v for k, v in optimizer_kwargs.items() if k in allowed[cfg.method]}

    if cfg.method == "de":
        optimizer_kwargs.setdefault(
            "tqdm_desc", f"Additive {cfg.n_partials}-osc DE → {str(cfg.metric).title()}"
        )
        result, history = run_de_optimization(obj, bounds, seed=cfg.seed, **optimizer_kwargs)
    elif cfg.method == "da":
        optimizer_kwargs.setdefault(
            "tqdm_desc", f"Additive {cfg.n_partials}-osc DA → {str(cfg.metric).title()}"
        )
        result, history = run_dual_annealing(obj, bounds, seed=cfg.seed, **optimizer_kwargs)
    elif cfg.method == "bh":
        optimizer_kwargs.setdefault(
            "tqdm_desc", f"Additive {cfg.n_partials}-osc BH → {str(cfg.metric).title()}"
        )
        result, history = run_basinhopping(obj, bounds, seed=cfg.seed, **optimizer_kwargs)
    else:
        raise ValueError(f"Unknown method: {cfg.method}")

    params = np.asarray(result.x, dtype=float)
    best = float(result.fun)

    # Filenames
    os.makedirs(cfg.audio_dir, exist_ok=True)
    os.makedirs(cfg.tsv_dir, exist_ok=True)
    os.makedirs(cfg.plots_dir, exist_ok=True)

    info = [str(cfg.method).strip().lower(), str(cfg.metric).strip().lower()]
    suffix = f"{'_'.join(info)}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    base_prefix = f"optimized_output_additive_{cfg.target_name}"

    # Audio
    wav_path = os.path.join(cfg.audio_dir, f"{base_prefix}_{suffix}.wav")

    _, y = additive_synth(
        params,
        duration=cfg.duration,
        sr=cfg.sr,
        waveforms=(cfg.waveforms if cfg.use_waveforms else None),
        dtype=np_dtype,
    )
    save_wav(
        wav_path,
        y,
        sr=cfg.sr,
        fade_in_ms=cfg.fade_in_ms,
        fade_out_ms=cfg.fade_out_ms,
        add_info=False,
        add_time=False,
    )

    # TSV
    tsv_path = os.path.join(cfg.tsv_dir, f"final_values_additive_{cfg.target_name}_{suffix}.tsv")
    _ = save_and_display_final_values_additive(
        params, tsv_path, waveforms=(cfg.waveforms if cfg.use_waveforms else None)
    )

    # Plots
    plot_paths: dict[str, str] = {}
    # time
    plt.figure(figsize=(10, 4))
    n_samples = min(1000, y.shape[0])
    plt.plot(y[:n_samples])
    plt.title("Optimized Combined Signal (time domain)")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    p_time = os.path.join(cfg.plots_dir, f"{base_prefix}_{suffix}_time.png")
    plt.savefig(p_time, dpi=150)
    plt.close()
    plot_paths["time"] = p_time
    print(f"[save_time_plot] wrote: {os.path.abspath(p_time)}")

    # spectrum
    freqs, mag = make_fft(y, sr=cfg.sr, fft_pad=cfg.fft_pad)
    mag = mag / (np.max(mag) + 1e-12)
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, mag, label="Optimized spectrum")
    if cfg.target_freqs is not None and cfg.target_amps is not None:
        t_amps = cfg.target_amps / (np.max(cfg.target_amps) + 1e-12)
        plt.scatter(cfg.target_freqs, t_amps, label="Target spectrum", color="red", zorder=5)
    plt.xlim(0, 8000)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (norm)")
    plt.title("Frequency Spectrum of Optimized Signal")
    plt.legend()
    plt.tight_layout()
    p_spec = os.path.join(cfg.plots_dir, f"{base_prefix}_{suffix}_spectrum.png")
    plt.savefig(p_spec, dpi=150)
    plt.close()
    plot_paths["spectrum"] = p_spec
    print(f"[save_spectrum_plot] wrote: {os.path.abspath(p_spec)}")

    # error history
    plt.figure(figsize=(10, 4))
    plt.plot(history)
    plt.title("Error progression")
    plt.xlabel("Iteration/Generation")
    plt.ylabel("Objective value")
    plt.grid(True)
    plt.tight_layout()
    p_err = os.path.join(cfg.plots_dir, f"{base_prefix}_{suffix}_error.png")
    plt.savefig(p_err, dpi=150)
    plt.close()
    plot_paths["error"] = p_err
    print(f"[save_error_plot] wrote: {os.path.abspath(p_err)}")

    return {
        "method": cfg.method,
        "metric": cfg.metric,
        "best": best,
        "params": params,
        "history": history,
        "wav_path": wav_path,
        "tsv_path": tsv_path,
        "plots": plot_paths,
    }


def _cli():
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Run one additive optimization from TSV target")
    parser.add_argument("--target-tsv", required=True, help="Path to TSV with columns: Frequency (Hz), Amplitude")
    parser.add_argument("--method", choices=["de", "da", "bh"], default="de")
    parser.add_argument(
        "--metric",
        default="pearson",
        choices=[
            "pearson",
            "mfcc",
            "itakura_saito",
            "spectral_convergence",
            "cosine",
            "euclidean",
            "manhattan",
            "kl",
        ],
    )
    parser.add_argument("--n-partials", type=int, default=4)
    parser.add_argument("--no-waveforms", action="store_true", help="Disable custom waveform set; use sines only")
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--fft-pad", type=int, default=2)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32", help="Waveform synthesis dtype")
    parser.add_argument("--fade-in-ms", type=float, default=10.0)
    parser.add_argument("--fade-out-ms", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audio-dir", default="rendered_audio")
    parser.add_argument("--plots-dir", default="rendered_plots")
    parser.add_argument("--tsv-dir", default="tsv")

    # Common optimizer kwargs (filtered per method later)
    parser.add_argument("--maxiter", type=int)
    parser.add_argument("--popsize", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--strategy")
    parser.add_argument("--tol", type=float)
    parser.add_argument("--mutation")  # allow string like "(0.5, 1.0)"
    parser.add_argument("--recombination", type=float)
    parser.add_argument("--initial-temp", type=float)
    parser.add_argument("--stepsize", type=float)

    args = parser.parse_args()

    df = pd.read_csv(args.target_tsv, sep="\t")
    target_freqs = df["Frequency (Hz)"].to_numpy()
    target_amps = (df["Amplitude"] / df["Amplitude"].max()).to_numpy()
    target_name = Path(args.target_tsv).stem

    # parse mutation if provided as string tuple
    mutation = args.mutation
    if isinstance(mutation, str):
        try:
            mutation_eval = eval(mutation, {"__builtins__": {}}, {})
            if isinstance(mutation_eval, (tuple, float)):
                mutation = mutation_eval
        except Exception:
            pass

    optimizer_kwargs = {}
    for k in [
        "maxiter",
        "popsize",
        "workers",
        "strategy",
        "tol",
        "recombination",
        "initial_temp",
        "stepsize",
    ]:
        v = getattr(args, k, None)
        if v is not None:
            optimizer_kwargs[k if k != "initial_temp" else "initial_temp"] = v
    if mutation is not None:
        optimizer_kwargs["mutation"] = mutation

    cfg = RunAdditiveConfig(
        target_freqs=target_freqs,
        target_amps=target_amps,
        target_name=target_name,
        n_partials=args.n_partials,
        use_waveforms=(not args.no_waveforms),
        waveforms=tuple(DEFAULT_WAVEFORMS),
        sr=args.sr,
        duration=args.duration,
        fft_pad=args.fft_pad,
        waveform_dtype=args.dtype,
        fade_in_ms=args.fade_in_ms,
        fade_out_ms=args.fade_out_ms,
        method=args.method,
        metric=args.metric,
        optimizer_kwargs=optimizer_kwargs,
        seed=args.seed,
        audio_dir=args.audio_dir,
        plots_dir=args.plots_dir,
        tsv_dir=args.tsv_dir,
    )

    info = run_one_additive(cfg)
    # Print concise summary
    print(
        f"Completed: method={info['method']} metric={info['metric']} best={info['best']:.6g}\n"
        f" wav={info['wav_path']}\n tsv={info['tsv_path']}\n plots={info['plots']}"
    )


if __name__ == "__main__":
    _cli()
