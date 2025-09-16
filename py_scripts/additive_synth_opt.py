from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr

# Reuse helpers from FM module for consistency (FFT, plotting, saving)
from .fm_synth_opt import (
    time_vector,
    sine_wave,
    make_fft,
    save_wav,
    plot_time,
    plot_spectrum,
)


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


def _quantize_index(x: float, n: int) -> int:
    if n <= 0:
        return 0
    i = int(np.round(float(x)))
    return int(np.clip(i, 0, n - 1))


def additive_synth(
    params: np.ndarray,
    duration: float,
    sr: int,
    *,
    waveforms: Optional[Sequence[Callable[[float, float, float, int], np.ndarray]]] = None,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Additive synthesis.

    If waveforms is None: params = [f1, a1, f2, a2, ...].
    If waveforms is provided: params = [f1, a1, w1, f2, a2, w2, ...] where w is an index.

    Returns (t, signal) normalized to max |x| <= 1.
    """
    t = time_vector(duration, sr, dtype=dtype)
    sig = np.zeros_like(t, dtype=dtype)
    p = np.asarray(params, dtype=float)

    if waveforms is None:
        assert p.size % 2 == 0, "Params must be [f1,a1,f2,a2,...] when waveforms is None"
        step = 2
    else:
        assert p.size % 3 == 0, "Params must be [f1,a1,w1,f2,a2,w2,...] when waveforms provided"
        step = 3

    for i in range(0, p.size, step):
        f = float(p[i])
        a = float(p[i + 1])
        if a <= 0.0 or f <= 0.0:
            continue
        if waveforms is None:
            sig += sine_wave(f, a, t, dtype=dtype)
        else:
            wi = _quantize_index(p[i + 2], len(waveforms))
            wav_fn = waveforms[wi]
            # waveform signature: (frequency, amplitude, duration, sample_rate)
            try:
                sig += wav_fn(f, a, duration, sr, dtype=dtype).astype(dtype)
            except TypeError:
                # Backward compatibility if waveform doesn't accept dtype
                sig += wav_fn(f, a, duration, sr).astype(dtype)
    maxv = float(np.max(np.abs(sig)))
    if maxv > 0:
        sig = sig / maxv
    return t, sig.astype(dtype)


def _place_kernel(
    target_freqs: np.ndarray,
    target_amps: np.ndarray,
    freqs: np.ndarray,
    kernel: Literal["none", "gaussian", "triangular"] = "gaussian",
    bw_hz: float = 2.0,
) -> np.ndarray:
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
            sigma = bw_hz / np.sqrt(8.0 * np.log(2.0))
            k = np.exp(-0.5 * ((freqs - f0) / max(sigma, 1e-6)) ** 2)
        else:  # triangular
            k = np.clip(1.0 - np.abs(freqs - f0) / max(bw_hz, 1e-6), 0.0, 1.0)
        out += (a0 * k).astype(np.float32)
    return out


@dataclass
class AdditiveObjective:
    target_freqs: np.ndarray
    target_amps: np.ndarray
    metric: MetricName = "pearson"
    n_partials: int = 4
    duration: float = 1.0
    sr: int = 44100
    fft_pad: int = 2
    target_kernel: Literal["none", "gaussian", "triangular"] = "gaussian"
    target_bw_hz: float = 2.0
    n_mfcc: int = 20
    seed: Optional[int] = 42
    waveforms: Optional[Sequence[Callable[[float, float, float, int], np.ndarray]]] = None
    dtype: np.dtype = np.float32

    # caches
    _freqs: Optional[np.ndarray] = None
    _target_spec: Optional[np.ndarray] = None
    _t: Optional[np.ndarray] = None
    _target_mfcc_mean: Optional[np.ndarray] = None

    def __post_init__(self):
        self.target_amps = self._normalize(self.target_amps)
        t_tmp = time_vector(self.duration, self.sr)
        freqs_tmp, _ = make_fft(np.zeros_like(t_tmp), self.sr, self.fft_pad, keep_dtype=True)
        self._freqs = freqs_tmp
        self._target_spec = _place_kernel(
            self.target_freqs.astype(np.float32),
            self.target_amps.astype(np.float32),
            self._freqs,
            kernel=self.target_kernel,
            bw_hz=self.target_bw_hz,
        ).astype(self.dtype)
        self._t = t_tmp

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        m = float(np.max(x)) if x.size else 0.0
        return x / m if m > 0 else x

    @property
    def n_params(self) -> int:
        return int(self.n_partials) * (3 if self.waveforms is not None else 2)

    def default_bounds(
        self,
        *,
        freq_lo: float = 5.0,
        freq_hi: float = 5000.0,
        amp_lo: float = 0.0,
        amp_hi: float = 1.0,
    ) -> list[tuple[float, float]]:
        b: list[tuple[float, float]] = []
        use_w = self.waveforms is not None
        for _ in range(self.n_partials):
            b.append((freq_lo, freq_hi))
            b.append((amp_lo, amp_hi))
            if use_w:
                # waveform index (continuous, will be quantized inside objective)
                n_w = len(self.waveforms or [])
                b.append((0.0, max(0.0, float(n_w - 1))))
        return b

    def _ensure_target_mfcc(self):
        if self._target_mfcc_mean is not None:
            return
        t = self._t if self._t is not None else time_vector(self.duration, self.sr)
        target_sig = np.zeros_like(t, dtype=np.float32)
        for f, a in zip(self.target_freqs, self.target_amps):
            target_sig += sine_wave(float(f), float(a), t)
        maxv = float(np.max(np.abs(target_sig)))
        if maxv > 0:
            target_sig = target_sig / maxv
        import librosa
        mfccs = librosa.feature.mfcc(y=target_sig, sr=self.sr, n_mfcc=self.n_mfcc)
        self._target_mfcc_mean = np.mean(mfccs, axis=1).astype(np.float32)

    def __call__(self, params: np.ndarray) -> float:
        # synthesize
        _, sig = additive_synth(
            params,
            duration=self.duration,
            sr=self.sr,
            waveforms=self.waveforms,
            dtype=self.dtype,
        )
        # spectrum
        _, mag = make_fft(sig, sr=self.sr, fft_pad=self.fft_pad, keep_dtype=True)
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
            # Safe cosine distance: handle zero-norm or non-finite values without warnings
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


def extract_frequencies_amplitudes(
    params: np.ndarray,
    *,
    waveforms: Optional[Sequence[Callable]] = None,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    p = np.asarray(params, dtype=float)
    if waveforms is None:
        assert p.size % 2 == 0
        freqs = p[0::2].copy()
        amps = p[1::2].copy()
        return freqs, amps, None
    else:
        assert p.size % 3 == 0
        freqs = p[0::3].copy()
        amps = p[1::3].copy()
        widx = p[2::3].copy()
        # quantize to valid integer indices
        if len(waveforms) > 0:
            widx = np.array([_quantize_index(x, len(waveforms)) for x in widx], dtype=int)
        else:
            widx = np.zeros_like(widx, dtype=int)
        return freqs, amps, widx


def save_and_display_final_values_additive(
    params: np.ndarray,
    tsv_filename: str = 'tsv/final_values_additive.tsv',
    *,
    waveforms: Optional[Sequence[Callable]] = None,
):
    import os
    import pandas as pd
    freqs, amps, widx = extract_frequencies_amplitudes(params, waveforms=waveforms)
    # Keep original order for mapping to oscillator index (1..N)
    data = {
        'Modulator': np.arange(1, len(freqs) + 1),
        'Frequency (Hz)': freqs,
        'Amplitude': amps,
    }
    if widx is not None and waveforms is not None and len(waveforms) > 0:
        names = []
        for i in widx.astype(int):
            try:
                nm = getattr(waveforms[i], '__name__', str(i))
            except Exception:
                nm = str(i)
            names.append(nm)
        data['Waveform'] = names
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(tsv_filename) or '.', exist_ok=True)
    df.to_csv(tsv_filename, sep='\t', index=False)
    print(f"TSV file was saved at: {os.path.abspath(tsv_filename)}")
    return df


# Re-export runners from FM module (they work with any callable objective)
from .fm_synth_opt import (
    run_de_optimization,
    run_dual_annealing,
    run_basinhopping,
)
