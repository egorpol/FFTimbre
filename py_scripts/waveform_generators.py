import numpy as np
from functools import lru_cache


def _dtype_to_str(dtype) -> str:
    try:
        return np.dtype(dtype).name
    except Exception:
        return "float32"


@lru_cache(maxsize=128)
def _cached_time_vector(n: int, duration: float, dtype_name: str):
    # endpoint=False to align with fm_synth_opt.time_vector
    return np.linspace(0.0, duration, n, endpoint=False, dtype=np.dtype(dtype_name))


def _time_vector(duration: float, sample_rate: int, dtype=np.float32) -> np.ndarray:
    n = int(round(sample_rate * duration))
    return _cached_time_vector(n, float(duration), _dtype_to_str(dtype))


def sine_wave(frequency, amplitude, duration, sample_rate, *, dtype=np.float32):
    t = _time_vector(float(duration), int(sample_rate), dtype=dtype)
    return (float(amplitude) * np.sin(2 * np.pi * float(frequency) * t)).astype(dtype)


def square_wave(frequency, amplitude, duration, sample_rate, *, dtype=np.float32):
    t = _time_vector(float(duration), int(sample_rate), dtype=dtype)
    return (float(amplitude) * np.sign(np.sin(2 * np.pi * float(frequency) * t))).astype(dtype)


def triangle_wave(frequency, amplitude, duration, sample_rate, *, dtype=np.float32):
    t = _time_vector(float(duration), int(sample_rate), dtype=dtype)
    return (
        float(amplitude)
        * (2.0 * np.abs(np.arcsin(np.sin(2 * np.pi * float(frequency) * t))) / np.pi)
    ).astype(dtype)


def sawtooth_wave(frequency, amplitude, duration, sample_rate, *, dtype=np.float32):
    t = _time_vector(float(duration), int(sample_rate), dtype=dtype)
    ft = t * float(frequency)
    return (float(amplitude) * (2.0 * (ft - np.floor(0.5 + ft)))).astype(dtype)


def noise_wave(frequency, amplitude, duration, sample_rate, *, dtype=np.float32):
    t = _time_vector(float(duration), int(sample_rate), dtype=dtype)
    # frequency unused; generate white noise
    return (float(amplitude) * np.random.normal(0.0, 1.0, size=t.shape).astype(dtype))
