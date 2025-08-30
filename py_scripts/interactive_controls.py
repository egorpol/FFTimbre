from __future__ import annotations

import os
import glob
from typing import Callable, Dict, Optional, Tuple, Any

import numpy as np
import ipywidgets as w
from IPython.display import display, Audio
import matplotlib.pyplot as plt


class InteractiveSpectralUI:
    """
    A lightweight, reusable ipywidgets control panel for spectral optimization
    workflows. Plug your existing notebook functions into this class to get a
    clean, user-focused interaction layer without scattering widget code across
    cells.

    Expected callback signatures (pass any you need; others can be None):

    - optimize_fn(settings: Dict[str, Any]) -> Dict[str, Any]
        Returns a result dict. Recommended keys: 'params', 'y', 'fs', 'log'.

    - synthesize_fn(settings: Dict[str, Any], params: Optional[np.ndarray])
        -> Tuple[np.ndarray, int]
        Returns audio samples y (float32/float64, -1..1) and sample rate fs.

    - compute_target_spectrum_fn(target_tsv_path: str)
        -> Tuple[np.ndarray, np.ndarray]
        Returns (freqs, amps) for plotting/verification.

    - save_wav_fn(y: np.ndarray, fs: int, filename: str) -> str
        Returns the saved file path.

    Notes
    -----
    - All callbacks are optional; UI will disable buttons if missing.
    - This class only coordinates UI and plotting; your notebook keeps the
      domain logic (optimization, FM/additive synthesis, etc.).
    """

    def __init__(
        self,
        *,
        mode: str = "fm",  # purely cosmetic label in the header
        optimize_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        synthesize_fn: Optional[
            Callable[[Dict[str, Any], Optional[np.ndarray]], Tuple[np.ndarray, int]]
        ] = None,
        compute_target_spectrum_fn: Optional[
            Callable[[str], Tuple[np.ndarray, np.ndarray]]
        ] = None,
        save_wav_fn: Optional[Callable[[np.ndarray, int, str], str]] = None,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.mode = mode
        self.optimize_fn = optimize_fn
        self.synthesize_fn = synthesize_fn
        self.compute_target_spectrum_fn = compute_target_spectrum_fn
        self.save_wav_fn = save_wav_fn
        self.defaults = defaults or {}

        self._last_result: Dict[str, Any] = {}
        self._last_audio: Optional[Tuple[np.ndarray, int]] = None

        # Build UI
        self.ui = self._build_ui()
        self._wire_events()

    # ---------------------------------------------------------------------
    # UI Construction
    # ---------------------------------------------------------------------
    def _build_ui(self) -> w.Widget:
        # Data/target selection
        tsv_files = sorted(glob.glob(os.path.join("tsv", "*.tsv")))
        xml_files = sorted(glob.glob(os.path.join("xml_presets", "*.xml")))
        adv_files = sorted(glob.glob(os.path.join("xml_presets", "*.adv")))

        # Normalize defaults to match option values across OS path separators
        _default_tsv = self.defaults.get("target_tsv", "")
        if _default_tsv:
            _default_tsv = os.path.normpath(_default_tsv)

        _default_preset = self.defaults.get("preset_path", "")
        if _default_preset:
            _default_preset = os.path.normpath(_default_preset)

        self.dd_tsv = w.Dropdown(
            options=[("Select target TSV…", "")] + [(os.path.basename(p), p) for p in tsv_files],
            value=_default_tsv or "",
            description="Target TSV",
            layout=w.Layout(width="100%"),
        )

        self.dd_preset = w.Dropdown(
            options=[("(optional)", "")] + [(os.path.basename(p), p) for p in (xml_files + adv_files)],
            value=_default_preset or "",
            description="Preset",
            layout=w.Layout(width="100%"),
        )

        # Core parameters
        self.int_osc = w.IntSlider(
            value=int(self.defaults.get("num_operators", 4)),
            min=1,
            max=16,
            step=1,
            description="# Ops",
            continuous_update=False,
        )

        self.float_duration = w.FloatSlider(
            value=float(self.defaults.get("duration", 2.0)),
            min=0.1,
            max=10.0,
            step=0.1,
            description="Dur (s)",
            continuous_update=False,
        )

        self.fs = w.Dropdown(
            options=[("44.1 kHz", 44100), ("48 kHz", 48000), ("88.2 kHz", 88200), ("96 kHz", 96000)],
            value=int(self.defaults.get("fs", 44100)),
            description="Fs",
        )

        # Optimization controls
        self.opt_method = w.Dropdown(
            options=["differential_evolution", "basinhopping", "dual_annealing"],
            value=str(self.defaults.get("optimizer", "differential_evolution")),
            description="Optimizer",
            layout=w.Layout(width="50%"),
        )

        self.int_iters = w.IntSlider(
            value=int(self.defaults.get("iterations", 200)),
            min=10,
            max=5000,
            step=10,
            description="Iters",
            continuous_update=False,
        )

        self.dd_metric = w.Dropdown(
            options=[
                ("Spectral Convergence", "spectral_convergence"),
                ("Cosine Similarity", "cosine_similarity"),
                ("Itakura-Saito", "itakura_saito"),
                ("Euclidean", "euclidean"),
                ("Manhattan", "manhattan"),
                ("Kullback-Leibler", "kullback_leibler"),
            ],
            value=str(self.defaults.get("metric", "spectral_convergence")),
            description="Metric",
            layout=w.Layout(width="50%"),
        )

        # Output controls
        self.txt_name = w.Text(
            value=str(self.defaults.get("output_name", "optimized_output.wav")),
            description="Output",
            layout=w.Layout(width="100%"),
        )

        # Action buttons
        self.btn_preview = w.Button(description="Preview Target", icon="line-chart")
        self.btn_optimize = w.Button(description="Optimize", button_style="warning", icon="cogs")
        self.btn_synthesize = w.Button(description="Synthesize", button_style="success", icon="play")
        self.btn_save = w.Button(description="Save WAV", icon="save")

        # Disable buttons if callbacks are missing
        self.btn_preview.disabled = self.compute_target_spectrum_fn is None
        self.btn_optimize.disabled = self.optimize_fn is None
        self.btn_synthesize.disabled = self.synthesize_fn is None
        self.btn_save.disabled = self.save_wav_fn is None

        # Output area
        self.out = w.Output(layout=w.Layout(border="1px solid #ddd"))

        # Layout
        header = w.HTML(f"<h3 style='margin:4px 0'>Spectral {self.mode.upper()} Control Panel</h3>")

        top = w.VBox([header, self.dd_tsv, self.dd_preset])

        row1 = w.HBox([self.int_osc, self.float_duration, self.fs])
        row2 = w.HBox([self.opt_method, self.dd_metric])
        row3 = w.HBox([self.int_iters])

        actions = w.HBox([self.btn_preview, self.btn_optimize, self.btn_synthesize, self.btn_save])

        body = w.VBox([row1, row2, row3, self.txt_name, actions, self.out])

        return w.VBox([top, body])

    def _wire_events(self) -> None:
        self.btn_preview.on_click(self._on_preview)
        self.btn_optimize.on_click(self._on_optimize)
        self.btn_synthesize.on_click(self._on_synthesize)
        self.btn_save.on_click(self._on_save)

    # ---------------------------------------------------------------------
    # Event handlers
    # ---------------------------------------------------------------------
    def _collect_settings(self) -> Dict[str, Any]:
        return {
            "target_tsv": self.dd_tsv.value or None,
            "preset_path": self.dd_preset.value or None,
            "num_operators": int(self.int_osc.value),
            "duration": float(self.float_duration.value),
            "fs": int(self.fs.value),
            "optimizer": self.opt_method.value,
            "metric": self.dd_metric.value,
            "iterations": int(self.int_iters.value),
            "output_name": self.txt_name.value,
            "mode": self.mode,
        }

    def _on_preview(self, _):
        if self.compute_target_spectrum_fn is None:
            return
        settings = self._collect_settings()
        tsv = settings["target_tsv"]
        if not tsv:
            with self.out:
                print("Please select a target TSV file.")
            return
        with self.out:
            self.out.clear_output()
            try:
                freqs, amps = self.compute_target_spectrum_fn(tsv)
                fig, ax = plt.subplots(figsize=(6, 3))
                # Matplotlib versions differ on 'use_line_collection'; keep compatibility
                try:
                    ax.stem(freqs, amps, basefmt=" ", use_line_collection=True)
                except TypeError:
                    ax.stem(freqs, amps, basefmt=" ")
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Amplitude")
                ax.set_title("Target Spectrum Preview")
                ax.grid(True, alpha=0.25)
                plt.show()
            except Exception as e:
                print(f"Preview error: {e}")

    def _on_optimize(self, _):
        if self.optimize_fn is None:
            return
        settings = self._collect_settings()
        with self.out:
            self.out.clear_output()
            print("Running optimization…")
            try:
                result = self.optimize_fn(settings)
                self._last_result = result or {}
                if result and "y" in result and "fs" in result:
                    y, fs = result["y"], int(result["fs"])
                    self._last_audio = (y, fs)
                    display(Audio(y, rate=fs))
                # Overlay plot: target spectrum vs optimized spectrum (red dots)
                try:
                    if self.compute_target_spectrum_fn is not None and settings.get("target_tsv"):
                        import numpy as np
                        tsv = settings["target_tsv"]
                        freqs_t, amps_t = self.compute_target_spectrum_fn(tsv)
                        # Compute optimized spectrum magnitudes at target freqs
                        if result and "y" in result and "fs" in result:
                            y_opt, fs_opt = result["y"], int(result["fs"])
                            if y_opt is not None and len(y_opt) > 0:
                                n = len(y_opt)
                                # Use rFFT for real signals
                                fft_mag = np.abs(np.fft.rfft(y_opt))
                                fft_freqs = np.fft.rfftfreq(n, 1.0 / fs_opt)
                                # Map target freqs to nearest FFT bins
                                mask = np.isfinite(freqs_t) & np.isfinite(amps_t) & (freqs_t <= fs_opt * 0.5) & (freqs_t >= 0)
                                freqs_t = freqs_t[mask]
                                amps_t = amps_t[mask]
                                if freqs_t.size > 0:
                                    idx = np.searchsorted(fft_freqs, freqs_t)
                                    idx = np.clip(idx, 0, fft_freqs.size - 1)
                                    opt_vals = fft_mag[idx]
                                    # Normalize both to unit max for visual comparison
                                    if np.max(amps_t) > 0:
                                        amps_t_n = amps_t / np.max(amps_t)
                                    else:
                                        amps_t_n = amps_t
                                    if np.max(opt_vals) > 0:
                                        opt_vals_n = opt_vals / np.max(opt_vals)
                                    else:
                                        opt_vals_n = opt_vals
                                    import matplotlib.pyplot as plt
                                    fig, ax = plt.subplots(figsize=(6, 3))
                                    try:
                                        ax.stem(freqs_t, amps_t_n, basefmt=" ", use_line_collection=True, label="target")
                                    except TypeError:
                                        ax.stem(freqs_t, amps_t_n, basefmt=" ", label="target")
                                    ax.scatter(freqs_t, opt_vals_n, color="red", s=14, label="optimized")
                                    ax.set_xlabel("Frequency (Hz)")
                                    ax.set_ylabel("Normalized Amplitude")
                                    ax.set_title("Target vs Optimized Spectrum")
                                    ax.grid(True, alpha=0.25)
                                    ax.legend(loc="best")
                                    plt.show()
                except Exception as e:
                    print(f"Overlay plot error: {e}")
                if result and "log" in result:
                    print(str(result["log"]))
                print("Optimization done.")
            except Exception as e:
                print(f"Optimization error: {e}")

    def _on_synthesize(self, _):
        if self.synthesize_fn is None:
            return
        settings = self._collect_settings()
        params = self._last_result.get("params") if self._last_result else None
        with self.out:
            self.out.clear_output()
            print("Synthesizing…")
            try:
                y, fs = self.synthesize_fn(settings, params)
                self._last_audio = (y, int(fs))
                display(Audio(y, rate=int(fs)))
                print("Synthesis done.")
            except Exception as e:
                print(f"Synthesis error: {e}")

    def _on_save(self, _):
        if self.save_wav_fn is None or self._last_audio is None:
            return
        y, fs = self._last_audio
        filename = self.txt_name.value or "optimized_output.wav"
        with self.out:
            try:
                path = self.save_wav_fn(y, int(fs), filename)
                print(f"Saved: {path}")
            except Exception as e:
                print(f"Save error: {e}")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def display(self) -> None:
        display(self.ui)

    def get_settings(self) -> Dict[str, Any]:
        return self._collect_settings()
