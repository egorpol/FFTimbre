from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

import numpy as np
from scipy.fft import rfft, rfftfreq, next_fast_len
from scipy.optimize import differential_evolution, dual_annealing, basinhopping
from scipy.spatial.distance import cosine as cosine_distance
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# -----------------------------------------------------------------------------
# Global context for automatic naming
# -----------------------------------------------------------------------------
# These are updated by optimization runners and used by save_wav when
# add_info=True but method/metric are not passed explicitly.
LAST_METHOD: Optional[str] = None  # e.g., "de", "da", "bh"
LAST_METRIC: Optional[str] = None  # e.g., "pearson", "mfcc"

# --- preview/save target as audio ---
def synthesize_target_additive(freqs, amps, duration, sr, fade_in_ms=10.0, fade_out_ms=10.0):
    n = int(np.round(duration * sr))
    t = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float32)
    y = np.zeros_like(t, dtype=np.float32)
    for f, a in zip(freqs, amps):
        if a <= 0:
            continue
        y += a * np.sin(2.0 * np.pi * float(f) * t)
    maxv = np.max(np.abs(y))
    if maxv > 0:
        y = y / maxv
    # apply independent fades (ms)
    y = apply_fade_in_out(y, sr, fade_in_ms=fade_in_ms, fade_out_ms=fade_out_ms)
    return t, y

# -----------------------------------------------------------------------------
# Synthesis helpers
# -----------------------------------------------------------------------------

def time_vector(duration: float, sr: int, *, dtype: np.dtype = np.float32) -> np.ndarray:
    n = int(np.round(duration * sr))
    # endpoint=False to avoid a discontinuity at t=duration
    return np.linspace(0.0, duration, n, endpoint=False, dtype=dtype)


def sine_wave(freq: float, amp: float, t: np.ndarray, *, dtype: np.dtype | None = None) -> np.ndarray:
    # Use dtype of provided time vector unless explicitly overridden
    if dtype is None:
        dtype = getattr(t, "dtype", np.float32)
    return (amp * np.sin(2.0 * np.pi * freq * t, dtype=dtype)).astype(dtype)


def fm_modulate(
    carrier_freq: float,
    carrier_amp: float,
    mod_signal: np.ndarray,
    t: np.ndarray,
    *,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    # "amp" acts as carrier amplitude; mod_signal is *phase* in radians
    if dtype is None:
        dtype = getattr(t, "dtype", np.float32)
    return (carrier_amp * np.sin(2.0 * np.pi * carrier_freq * t + mod_signal, dtype=dtype)).astype(dtype)


def synth_chain(
    params: np.ndarray,
    duration: float,
    sr: int,
    *,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    4-osc FM chain (osc4 -> osc3 -> osc2 -> osc1):
    params = [f4, a4, f3, a3, f2, a2, f1, a1]
    Returns (t, signal) normalized to max |x| <= 1.
    """
    t = time_vector(duration, sr, dtype=dtype)
    f4, a4, f3, a3, f2, a2, f1, a1 = params
    mod4 = sine_wave(f4, a4, t, dtype=dtype)
    mod3 = fm_modulate(f3, a3, mod4, t, dtype=dtype)
    mod2 = fm_modulate(f2, a2, mod3, t, dtype=dtype)
    sig = fm_modulate(f1, a1, mod2, t, dtype=dtype)

    maxv = np.max(np.abs(sig))
    if maxv > 0:
        sig = sig / maxv
    return t, sig.astype(dtype)

# -----------------------------------------------------------------------------
# Spectral helpers
# -----------------------------------------------------------------------------

def make_fft(
    signal: np.ndarray,
    sr: int,
    fft_pad: int = 1,
    *,
    keep_dtype: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (freqs, mag) using Hann window + rFFT with zero-padding factor fft_pad.

    If keep_dtype is True, magnitude array matches the dtype of ``signal``; otherwise float32.
    """
    n = len(signal)
    n_fft = next_fast_len(n * max(1, int(fft_pad)))
    window = np.hanning(n).astype(np.float32)
    win_energy = np.sum(window**2) / n
    xw = signal[:n] * window
    spec = rfft(xw, n=n_fft)
    out_dtype = signal.dtype if keep_dtype else np.float32
    mag = np.abs(spec).astype(out_dtype)
    # Optional window normalization (so amplitude isn't arbitrarily reduced)
    if win_energy > 0:
        mag = mag / np.sqrt(win_energy)
    freqs = rfftfreq(n_fft, 1.0 / sr)
    return freqs, mag


def _place_kernel(target_freqs: np.ndarray,
                  target_amps: np.ndarray,
                  freqs: np.ndarray,
                  kernel: Literal["none", "gaussian", "triangular"] = "gaussian",
                  bw_hz: float = 2.0) -> np.ndarray:
    """
    Project discrete (f, A) into the rFFT bin grid with a small spread, which
    makes objectives smoother and avoids zero-bin divisions.
    """
    out = np.zeros_like(freqs, dtype=np.float32)
    if kernel == "none" or bw_hz <= 0:
        # nearest-bin placement
        idx = np.abs(freqs[:, None] - target_freqs[None, :]).argmin(axis=0)
        np.add.at(out, idx, target_amps.astype(np.float32))
        return out

    for f0, a0 in zip(target_freqs, target_amps):
        if a0 <= 0:
            continue
        if kernel == "gaussian":
            sigma = bw_hz / np.sqrt(8.0 * np.log(2.0))  # FWHM to sigma
            k = np.exp(-0.5 * ((freqs - f0) / max(sigma, 1e-6)) ** 2)
        else:  # triangular
            k = np.clip(1.0 - np.abs(freqs - f0) / max(bw_hz, 1e-6), 0.0, 1.0)
        out += (a0 * k).astype(np.float32)
    return out

# -----------------------------------------------------------------------------
# Objective
# -----------------------------------------------------------------------------

MetricName = Literal[
    "itakura_saito",
    "spectral_convergence",
    "cosine",
    "euclidean",
    "manhattan",
    "kl",
    "pearson",
    "mfcc",
]

@dataclass
class FMObjective:
    target_freqs: np.ndarray
    target_amps: np.ndarray
    metric: MetricName = "pearson"
    duration: float = 1.0
    sr: int = 44100
    fft_pad: int = 1
    target_kernel: Literal["none", "gaussian", "triangular"] = "gaussian"
    target_bw_hz: float = 2.0
    n_mfcc: int = 20
    seed: Optional[int] = 42
    dtype: np.dtype = np.float32

    # internal caches
    _t: Optional[np.ndarray] = None
    _target_spec: Optional[np.ndarray] = None
    _target_mfcc_mean: Optional[np.ndarray] = None
    _freqs: Optional[np.ndarray] = None

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        _ = rng.random()  # touch RNG for reproducibility elsewhere if desired
        self.target_amps = self._normalize(self.target_amps)
        # prepare representations that match model FFT resolution
        t_tmp = time_vector(self.duration, self.sr, dtype=self.dtype)
        # Use the frequency grid returned by make_fft to ensure
        # the target spectrum matches synthesized spectra length.
        freqs_tmp, _ = make_fft(np.zeros_like(t_tmp), self.sr, self.fft_pad, keep_dtype=True)
        self._freqs = freqs_tmp
        self._target_spec = _place_kernel(
            self.target_freqs.astype(np.float32),
            self.target_amps.astype(np.float32),
            self._freqs,
            kernel=self.target_kernel,
            bw_hz=self.target_bw_hz,
        ).astype(self.dtype)
        # target MFCC from additive sines (optional, only computed if needed)
        self._t = t_tmp

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        m = np.max(x)
        return x / m if m > 0 else x

    @property
    def n_params(self) -> int:
        return 8  # four oscillators (f, a) each

    def _ensure_target_mfcc(self):
        if self._target_mfcc_mean is not None:
            return
        t = self._t if self._t is not None else time_vector(self.duration, self.sr, dtype=self.dtype)
        # build an additive target signal from partials for MFCC comparison
        target_sig = np.zeros_like(t, dtype=np.float32)
        for f, a in zip(self.target_freqs, self.target_amps):
            target_sig += sine_wave(float(f), float(a), t)
        maxv = np.max(np.abs(target_sig))
        if maxv > 0:
            target_sig = target_sig / maxv
        # librosa import is done lazily to keep import time low if unused
        import librosa
        mfccs = librosa.feature.mfcc(y=target_sig, sr=self.sr, n_mfcc=self.n_mfcc)
        self._target_mfcc_mean = np.mean(mfccs, axis=1).astype(np.float32)

    def __call__(self, params: np.ndarray) -> float:
        # synthesize
        _, sig = synth_chain(params, duration=self.duration, sr=self.sr, dtype=self.dtype)
        # spectrum
        freqs, mag = make_fft(sig, sr=self.sr, fft_pad=self.fft_pad, keep_dtype=True)
        # align to cached grid (should already match)
        target = self._target_spec
        eps = 1e-12

        if self.metric == "itakura_saito":
            ratio = (target + eps) / (mag + eps)
            return float(np.sum(ratio - np.log(ratio) - 1.0))

        if self.metric == "spectral_convergence":
            num = np.linalg.norm(mag - target)
            den = np.linalg.norm(target) + eps
            return float(num / den)

        if self.metric == "cosine":
            # Safe cosine distance: guard against zero vectors to avoid divide-by-zero warnings
            norm_mag = float(np.linalg.norm(mag))
            norm_tgt = float(np.linalg.norm(target))
            denom = norm_mag * norm_tgt
            if denom <= 1e-12 or not np.isfinite(denom):
                return 1.0
            num = float(np.dot(mag, target))
            val = 1.0 - (num / denom)
            if not np.isfinite(val):
                return 1.0
            return float(val)

        if self.metric == "euclidean":
            return float(np.linalg.norm(mag - target))

        if self.metric == "manhattan":
            return float(np.sum(np.abs(mag - target)))

        if self.metric == "kl":
            p = target / (np.sum(target) + eps)
            q = mag / (np.sum(mag) + eps)
            return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))

        if self.metric == "pearson":
            # Guard against constant input which triggers scipy ConstantInputWarning
            if float(np.std(mag)) < 1e-12 or float(np.std(target)) < 1e-12:
                return 1.0
            r, _ = pearsonr(mag, target)
            if not np.isfinite(r):
                return 1.0
            return float(1.0 - r)

        if self.metric == "mfcc":
            self._ensure_target_mfcc()
            import librosa
            mfccs = librosa.feature.mfcc(y=sig, sr=self.sr, n_mfcc=self.n_mfcc)
            gen_mean = np.mean(mfccs, axis=1).astype(np.float32)
            return float(np.linalg.norm(gen_mean - self._target_mfcc_mean))

        raise ValueError(f"Unknown metric: {self.metric}")

    # Convenience: bounds factory
    def default_bounds(self, freq_lo: float = 50.0, freq_hi: float = 5000.0,
                       amp_lo: float = 0.0, amp_hi: float = 10.0) -> list[tuple[float, float]]:
        return [
            (freq_lo, freq_hi), (amp_lo, amp_hi),  # osc4
            (freq_lo, freq_hi), (amp_lo, amp_hi),  # osc3
            (freq_lo, freq_hi), (amp_lo, amp_hi),  # osc2
            (freq_lo, freq_hi), (amp_lo, amp_hi),  # osc1
        ]

# -----------------------------------------------------------------------------
# Optimization runners
# -----------------------------------------------------------------------------

def run_de_optimization(
    objective: FMObjective,
    bounds: list[tuple[float, float]],
    *,
    maxiter: int = 600,
    popsize: int = 15,
    tol: float = 1e-6,
    mutation: tuple[float, float] | float = (0.5, 1.0),
    recombination: float = 0.7,
    strategy: str = "best1bin",
    workers: int | None = None,
    seed: Optional[int] = 42,
    tqdm_desc: Optional[str] = None,
):
    """Run SciPy differential_evolution with a nice progress bar.

    Returns (result, error_history).
    """
    # Record method/metric for downstream filename annotation
    global LAST_METHOD, LAST_METRIC
    LAST_METHOD = "de"
    try:
        LAST_METRIC = str(objective.metric)
    except Exception:
        LAST_METRIC = None

    error_history: list[float] = []
    desc = tqdm_desc or f"FM 4-osc DE → {str(objective.metric).title()}"
    bar = tqdm(total=maxiter, unit="gen", desc=desc,
               bar_format='{l_bar}{bar} {n_fmt}/{total_fmt} [ETA: {remaining}, Elapsed: {elapsed}]')

    def cb(xk, convergence):
        val = float(objective(np.asarray(xk)))
        error_history.append(val)
        bar.update(1)
        return False  # keep going

    kwargs = dict(
        strategy=strategy,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        mutation=mutation,
        recombination=recombination,
        callback=cb,
        seed=seed,
    )
    if workers is not None and workers != 1:
        kwargs.update(dict(workers=workers, updating="deferred"))

    result = differential_evolution(objective, bounds, **kwargs)
    bar.close()
    return result, error_history


def run_dual_annealing(
    objective: FMObjective,
    bounds: list[tuple[float, float]],
    *,
    maxiter: int = 600,
    initial_temp: float = 5230.0,
    seed: Optional[int] = 42,
    tqdm_desc: Optional[str] = None,
):
    """Run SciPy dual_annealing with a nice progress bar.

    Returns (result, error_history).
    """
    # Record method/metric for downstream filename annotation
    global LAST_METHOD, LAST_METRIC
    LAST_METHOD = "da"
    try:
        LAST_METRIC = str(objective.metric)
    except Exception:
        LAST_METRIC = None

    error_history: list[float] = []
    desc = tqdm_desc or f"FM 4-osc DA → {str(objective.metric).title()}"
    bar = tqdm(total=maxiter, unit="iter", desc=desc,
               bar_format='{l_bar}{bar} {n_fmt}/{total_fmt} [ETA: {remaining}, Elapsed: {elapsed}]')

    def cb(x, f, context):
        # SciPy calls this each iteration; return True to stop
        val = float(objective(np.asarray(x)))
        error_history.append(val)
        bar.update(1)
        return False

    result = dual_annealing(objective, bounds,
                            maxiter=maxiter,
                            initial_temp=initial_temp,
                            seed=seed,
                            callback=cb)
    bar.close()
    return result, error_history

def run_basinhopping(
    objective: FMObjective,
    bounds: list[tuple[float, float]],
    *,
    maxiter: int = 200,
    stepsize: float = 0.5,
    seed: Optional[int] = 42,
    x0: Optional[np.ndarray] = None,
    tqdm_desc: Optional[str] = None,
):
    """Run SciPy basinhopping with bounds and a progress bar.

    Returns (result, error_history).
    """
    # Record method/metric for downstream filename annotation
    global LAST_METHOD, LAST_METRIC
    LAST_METHOD = "bh"
    try:
        LAST_METRIC = str(objective.metric)
    except Exception:
        LAST_METRIC = None

    rng = np.random.default_rng(seed)
    if x0 is None:
        # start at bounds midpoints
        lows = np.array([b[0] for b in bounds], dtype=float)
        highs = np.array([b[1] for b in bounds], dtype=float)
        x0 = (lows + highs) / 2.0

    class ClipStep:
        def __init__(self, bounds, stepsize, rng):
            self.bounds = np.array(bounds, dtype=float)
            self.stepsize = float(stepsize)
            self.rng = rng

        def __call__(self, x):
            x = np.array(x, dtype=float)
            step = self.rng.normal(scale=self.stepsize, size=x.shape)
            x_new = x + step
            # clip to bounds to remain feasible
            lows = self.bounds[:, 0]
            highs = self.bounds[:, 1]
            return np.clip(x_new, lows, highs)

    error_history: list[float] = []
    desc = tqdm_desc or f"FM 4-osc BH → {str(objective.metric).title()}"
    bar = tqdm(total=maxiter, unit="iter", desc=desc,
               bar_format='{l_bar}{bar} {n_fmt}/{total_fmt} [ETA: {remaining}, Elapsed: {elapsed}]')

    def cb(x, f, accept):
        error_history.append(float(f))
        bar.update(1)

    minimizer_kwargs = {
        'method': 'L-BFGS-B',
        'bounds': bounds,
    }

    result = basinhopping(
        objective, x0,
        niter=maxiter,
        stepsize=stepsize,
        take_step=ClipStep(bounds, stepsize, rng),
        minimizer_kwargs=minimizer_kwargs,
        callback=cb,
        seed=seed,
        disp=False,
    )
    bar.close()
    return result, error_history

# -----------------------------------------------------------------------------
# Plotting helpers (kept minimal / no seaborn)
# -----------------------------------------------------------------------------

def plot_time(signal: np.ndarray, sr: int, n_samples: int = 1000, show: bool = True):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(signal[:n_samples])
    ax.set_title("Optimized Combined Signal (time domain)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    if show:
        plt.show()
        plt.close(fig)
    else:
        # Close to avoid duplicated inline display (open-figure + returned fig)
        plt.close(fig)
    return fig


def plot_spectrum(
    signal: np.ndarray,
    sr: int,
    target_freqs: Optional[np.ndarray] = None,
    target_amps: Optional[np.ndarray] = None,
    xlim: Optional[Tuple[float, float]] = None,
    fft_pad: int = 1,
    show: bool = True,
):
    freqs, mag = make_fft(signal, sr=sr, fft_pad=fft_pad)
    mag = mag / (np.max(mag) + 1e-12)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs, mag, label="Optimized spectrum")
    if target_freqs is not None and target_amps is not None:
        t_amps = target_amps / (np.max(target_amps) + 1e-12)
        ax.scatter(target_freqs, t_amps, label="Target spectrum", color='red', zorder=5)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (norm)")
    ax.set_title("Frequency Spectrum of Optimized Signal")
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
        plt.close(fig)
    else:
        # Close to avoid duplicated inline display (open-figure + returned fig)
        plt.close(fig)
    return fig


def plot_error_history(history: list[float]):
    plt.figure(figsize=(10, 4))
    plt.plot(history)
    plt.title("Error progression")
    plt.xlabel("Iteration/Generation")
    plt.ylabel("Objective value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def apply_fade_in_out(signal: np.ndarray, sr: int, *, fade_in_ms: float = 0.0, fade_out_ms: float = 0.0) -> np.ndarray:
    """Return a copy of signal with fade-in/out applied using linear ramps.

    Durations are specified in milliseconds. Each fade is clamped to n//2 samples.
    """
    y = np.array(signal, dtype=np.float32, copy=True)
    n = y.shape[0]
    if n == 0:
        return y
    # compute fade sample counts and clamp
    n_fade_in = int(round(sr * max(0.0, float(fade_in_ms)) / 1000.0))
    n_fade_out = int(round(sr * max(0.0, float(fade_out_ms)) / 1000.0))
    if n_fade_in > 0:
        n_fade_in = min(n_fade_in, n // 2)
        if n_fade_in > 0:
            fade = np.linspace(0.0, 1.0, n_fade_in, dtype=np.float32)
            y[:n_fade_in] *= fade
    if n_fade_out > 0:
        n_fade_out = min(n_fade_out, n // 2)
        if n_fade_out > 0:
            fade = np.linspace(1.0, 0.0, n_fade_out, dtype=np.float32)
            y[-n_fade_out:] *= fade
    return y


def save_wav(
    path: str,
    signal: np.ndarray,
    sr: int = 44100,
    *,
    fade_in_ms: float = 0.0,
    fade_out_ms: float = 0.0,
    add_info: bool = False,
    add_time: bool = False,
    method: str | None = None,
    metric: str | None = None,
):
    """Lightweight WAV writer (int16). Creates directories if needed.

    - Applies optional linear fade-in/out in milliseconds before normalization.
    - Optionally augments the output filename with optimization info and timestamp.

    If ``add_info`` is True and either ``method`` or ``metric`` is provided,
    the base filename is suffixed with ``_<method>_<metric>`` (omitting any that
    are None/empty). If ``add_time`` is True, a ``_YYYYMMDD-HHMMSS`` suffix is
    added as well.
    """
    import os
    from datetime import datetime
    from scipy.io import wavfile

    # derive final output path with optional info/time suffixes
    directory, filename = os.path.split(path)
    base, ext = os.path.splitext(filename or "output.wav")

    suffix_parts: list[str] = []
    if add_info:
        # Auto-fill from last recorded method/metric if not provided
        if method is None:
            method = LAST_METHOD
        if metric is None:
            metric = LAST_METRIC
        info_parts: list[str] = []
        if method:
            info_parts.append(str(method).strip().lower())
        if metric:
            info_parts.append(str(metric).strip().lower())
        if info_parts:
            suffix_parts.append("_".join(info_parts))
    if add_time:
        suffix_parts.append(datetime.now().strftime("%Y%m%d-%H%M%S"))

    final_filename = base
    if suffix_parts:
        final_filename += "_" + "_".join(suffix_parts)
    final_filename += ext or ".wav"

    final_path = os.path.join(directory or ".", final_filename)

    # ensure directory exists
    os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)

    # prepare audio
    x = np.asarray(signal, dtype=np.float32)
    x = apply_fade_in_out(x, sr, fade_in_ms=fade_in_ms, fade_out_ms=fade_out_ms)
    x = x / (np.max(np.abs(x)) + 1e-12)

    # write wav
    wavfile.write(final_path, sr, (x * 32767.0).astype(np.int16))
    print(f"[save_wav] wrote: {os.path.abspath(final_path)}")
    return final_path

def extract_frequencies_amplitudes(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (frequencies, amplitudes) as arrays [f4, f3, f2, f1], [a4, a3, a2, a1]."""
    p = np.asarray(params, dtype=float)
    f4, a4, f3, a3, f2, a2, f1, a1 = p
    freqs = np.array([f4, f3, f2, f1], dtype=float)
    amps = np.array([a4, a3, a2, a1], dtype=float)
    return freqs, amps

def save_and_display_final_values(params: np.ndarray, tsv_filename: str = 'tsv/final_values_fm.tsv'):
    """Create a DataFrame with final values, display it, and save as TSV.

    Follows the format provided in the user's example code.
    Returns the pandas DataFrame.
    """
    import os
    import pandas as pd
    from IPython.display import display

    freqs, amps = extract_frequencies_amplitudes(params)
    df = pd.DataFrame({
        'Modulator': [1, 2, 3, 4],
        'Frequency (Hz)': freqs[::-1],
        'Amplitude': amps[::-1],
    })
    # display(df)

    os.makedirs(os.path.dirname(tsv_filename) or '.', exist_ok=True)
    df.to_csv(tsv_filename, sep='\t', index=False)
    print(f"TSV file was saved at: {os.path.abspath(tsv_filename)}")
    return df
