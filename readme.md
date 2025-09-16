# FFTimbre

[![License](https://img.shields.io/github/license/egorpol/FFTimbre)](LICENSE) [![Last commit](https://img.shields.io/github/last-commit/egorpol/FFTimbre)](https://github.com/egorpol/FFTimbre/commits) [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](#installation) [![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange)](https://jupyter.org)

FFTimbre is a small toolkit of notebooks and helpers for spectral matching via FM and additive synthesis. It started as an exploratory idea inside [audiospylt](https://github.com/egorpol/audiospylt) and has since grown into its own project.

Live site (audio + plots + evaluation): https://egorpol.github.io/FFTimbre/

This work was presented at the InMusic'24 conference (Oslo). Slides are available in `inmusic24_slides/`. A paper version is planned for the InMusic'24 proceedings (Routledge).

## Overview

The core question: how can we reproduce a target spectrum with a constrained oscillator setup? The initial motivation was to automatically derive settings for a Doepfer A-100 modular configuration (mostly for fun), but it has evolved into a general tool for resynthesizing spectra, with export to Ableton Operator in mind.

The approach is inspired by ideas from Risset & Wessel (1999), where a human judges the “goodness of fit” by comparing synthesized and original sounds. Here, we explore automated evaluation via spectral metrics and global optimizers.

![Risset/Wessel concept](risset.png)

## General concept

- We take a table of frequency/amplitude pairs (e.g., from [audiospylt](https://github.com/egorpol/audiospylt)) as the target.
- We compare target and synthesized spectra using multiple similarity metrics:
  - Itakura-Saito, Spectral Convergence, Cosine, Euclidean, Manhattan, KL, Pearson, MFCC distance.
- We optimize synthesis parameters using multiple global optimizers:
  - Differential Evolution (DE), Dual Annealing (DA), Basin Hopping (BH).
A graphical overview:

![Workflow schema](schema.png)

After optimization we get a TSV with oscillator values. We can synthesize it in Python right away or export it as an Ableton Operator preset.
We target two Operator setups: FM (algorithm 0 - the leftmost in the Operator GUI) and additive (algorithm 10 - the rightmost).
To go from a TSV of optimized oscillator settings to the Operator preset, we adjust value formatting and write the preset data into a compressed XML file.

In its current state, FFTimbre is an exploration tool that showcases various metrics and optimizers, aimed at artistic and exploratory use. Some metric/optimizer combinations can produce unexpected yet interesting results, including unconventional parameter settings. A general limitation is that it currently operates on a single DFT frame, which restricts the system to static sounds.

An improved version with additional metrics and optimizers, as well as a freely configurable oscillator setup, is currently in development and planned for release next year.

## Repository Structure

- `inmusic24_slides/` - conference slides (PDF/PPTX)
- Website (GitHub Pages)
  - `index.md` - site homepage with audio/plot gallery
  - `_config.yml` - Jekyll site config (project page)
  - `_includes/` - reusable HTML includes
    - `sample.html` - audio + plot block
    - `tsv_table.html` - dynamic TSV preview
  - `assets/style.css` - site stylesheet
  - `examples/` - per-example subpages and evaluation explorers (cello/voice/Parmegiani)
    - `*_eval.csv` and `*_eval.md` feed the Jekyll evaluation explorers.
- `py_scripts/`
  - `fm_synth_opt.py` - objective, synthesis, optimization runners, plotting, I/O helpers
  - `optimization_workflow.py` - batch job utilities and consistent file naming
  - `objective_functions.py` - legacy/experimental objective prototypes
  - `generate_wave_file.py` - WAV rendering at various sample rates/bit depths
  - `interactive_controls.py` - reusable ipywidgets UI for notebook workflows
  - `waveform_generators.py` - basic waveform functions (sine/square/triangle/saw/noise)
- `tsv/` - example target partials and saved final oscillator values (e.g., `cello_single.tsv`, `noanoa_single.tsv`, `final_values_fm_*.tsv`)
- `xml_presets/` - example Operator preset files (`.adv` Ableton presets and `.xml` sources)
- `rendered_audio/` - generated WAV files
- `rendered_gifs/` - optimizer animations
- `rendered_plots/` - saved PNG plots from runs

- `risset.png`, `schema.png` - figures used in the README

- Notebooks
  - `distances_demo.ipynb` - overview of objective functions on a simple spectrum
  - `optimization_gif.ipynb` - preview of the optimization algorithm workflow
  - `parameter_operator_fm.ipynb`, `parameter_operator_additive.ipynb` - parameter mapping
  - `preset_editor_operator_fm.ipynb`, `preset_editor_operator_additive.ipynb` - preset export/editing
  - `spectral_ml_additive3.ipynb` - earlier additive synthesis experiments
  - `spectral_ml_additive4.ipynb` - latest additive single-run workflow (optimize -> preview -> save)
  - `spectral_ml_additive4_batch.ipynb` - additive batch runner for metrics/optimizers with consistent naming
  - `spectral_ml_fm3.ipynb`, `spectral_ml_fm3_interactive.ipynb` - earlier/interactive FM experiments
  - `spectral_ml_fm4.ipynb` - guided single-run workflow (optimize -> preview -> save)
  - `spectral_ml_fm4_batch.ipynb` - batch runner for metrics/optimizers with consistent naming
  - `evaluate_cello_examples.ipynb`, `evaluate_parm_examples.ipynb`, `evaluate_voice_examples.ipynb` - generate evaluation CSVs and audio links for the site explorers. Includes an interactive IPython tool for visualization and exploration (prototype for the live-page implementation).
  - `render_from_tsv_fm.ipynb` - render audio from saved TSV with optimized FM values
  - `render_from_tsv_table.ipynb` - render audio from TSV of partials (freq/amp)

## Installation

Prerequisites: Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

## Quick Start

- Install dependencies (above).
- Open `spectral_ml_fm4.ipynb` and run the cells.
- Pick a target from `tsv/` (e.g., `cello_single.tsv`), run optimization, and preview.
- Outputs are written to `rendered_audio/` (WAV), `rendered_plots/` (PNG), and `tsv/` (final values).
- For multiple metrics/optimizers, try `spectral_ml_fm4_batch.ipynb`.

Optimization runs are slow: a single optimization can take 20-30 minutes. Lowering `WAVEFORM_DTYPE` to `'float32'` in the notebooks trades precision for speed compared to `'float64'`, but full batch runs may still take hours on an 8 vCPU machine (only Differential Evolution benefits from multiple workers).

## GitHub Pages (Site)

- **Hosting**: Built with Jekyll and published via GitHub Pages as a project site at `https://egorpol.github.io/FFTimbre/`.
- **Config**: `/_config.yml` sets `url` and `baseurl` for project hosting.
- **Homepage**: `/index.md` shows cello results plus links to other examples.
- **Examples**: Per-example subpages live in `/examples/`:
  - `/examples/cello.md`
  - `/examples/voice-single2.md`
  - `/examples/parm.md`
- **Evaluation explorers**: `/examples/*_eval.md` embed `csv_explorer.html` to browse the exported metric tables.
- **Includes**: Reusable blocks in `/_includes/`:
  - `tsv_table.html` renders TSV previews (client-side).
  - `sample.html` embeds audio and one or two plots side-by-side.
- **Clickable plots**: All plots are clickable to open the full-resolution image in a new tab.
- **Anchors**: Section IDs come from titles (slugified), so legend links like "Optimized FM with DE + cosine" -> `#optimized-fm-with-de-cosine` work automatically.

## Evaluation Workflows

Evaluation notebooks export the metric tables consumed by the GitHub Pages explorers.

- `evaluate_cello_examples.ipynb`, `evaluate_parm_examples.ipynb`, `evaluate_voice_examples.ipynb` aggregate optimization runs, export `examples/*_eval.csv`, and stage audio references in `rendered_audio/`.
- The generated CSV files pair with `examples/*_eval.md`, which wrap the `_includes/csv_explorer.html` component to produce interactive tables.
- Each explorer surfaces normalized metrics alongside direct audio previews, mirroring the columns captured in the CSV output.

## Usage (Programmatic)

Minimal FM optimization from Python, using helpers in `py_scripts/`:

```python
import numpy as np
import pandas as pd

from py_scripts.fm_synth_opt import (
    FMObjective,
    run_de_optimization,
    synth_chain,
    save_wav,
    save_and_display_final_values,
)

# 1) Load target spectrum (TSV with columns: Modulator, Frequency (Hz), Amplitude)
df = pd.read_csv("tsv/cello_single.tsv", sep="\t")
target_freqs = df["Frequency (Hz)"].to_numpy(dtype=float)
target_amps = df["Amplitude"].to_numpy(dtype=float)

# 2) Define objective
obj = FMObjective(
    target_freqs=target_freqs,
    target_amps=target_amps,
    metric="pearson",  # or: 'mfcc', 'itakura_saito', 'kl', ...
    duration=1.0,
    sr=44100,
    fft_pad=2,
    seed=42,
)
bounds = obj.default_bounds(freq_lo=5.0, freq_hi=5000.0, amp_lo=0.0, amp_hi=10.0)

# 3) Optimize (Differential Evolution)
result, history = run_de_optimization(obj, bounds, maxiter=300, workers=-1)
params = np.asarray(result.x, dtype=float)

# 4) Synthesize and save
_, y = synth_chain(params, duration=obj.duration, sr=obj.sr)
save_wav("rendered_audio/optimized_output_fm.wav", y, sr=obj.sr, add_info=True, add_time=True)
save_and_display_final_values(params, "tsv/final_values_fm.tsv")
```

## Metrics and Optimizers

- **Objective metrics**: Itakura-Saito (`itakura_saito`), Spectral Convergence (`spectral_convergence`), Cosine (`cosine`), Euclidean (`euclidean`), Manhattan (`manhattan`), KL (`kl`), Pearson (`pearson`), MFCC (`mfcc`).
- **Optimizers**: Differential Evolution (DE), Dual Annealing (DA), Basin Hopping (BH).

## Evaluation metrics (lower is better unless noted)
- Time MSE: Mean-squared error in time domain after RMS normalization and short-window alignment; $\mathrm{MSE}(x,y) = \tfrac{1}{N}\sum_{i=1}^{N}(x_i - y_i)^2$.
- Cosine distance (log-magnitude STFT): Cosine distance of flattened log-magnitude STFTs; $1 - \tfrac{\langle a, b \rangle}{\|a\|\,\|b\|}$, where $a = \mathrm{vec}(\log|S|)$ and $b = \mathrm{vec}(\log|\hat S|)$.
- Pearson distance (log-magnitude STFT): $1 - \rho(a,b)$, with $a = \mathrm{vec}(\log|S|)$ and $b = \mathrm{vec}(\log|\hat S|)$.
- Spectral convergence: $\tfrac{\|S - \hat S\|_F}{\|S\|_F + \varepsilon}$; emphasizes relative spectral errors.
- Log-spectral distance (dB): RMSE between $20\log_{10}|S|$ and $20\log_{10}|\hat S|$.
- Itakura-Saito divergence (power): $D_{\mathrm{IS}}(P \Vert \hat P) = \sum\!\left(\tfrac{P}{\hat P} - \log\tfrac{P}{\hat P} - 1\right)$, with $P = |S|^2$.
- MFCC L2: $\|M - \hat M\|_2$ for MFCC sequences (aligned to the minimum length).
- Spectral flatness L1: Mean absolute difference of spectral flatness over time; $\tfrac{1}{T}\sum_t |\mathrm{SFM}_t - \widehat{\mathrm{SFM}}_t|$.
- Centroid RMSE (Hz): $\sqrt{\tfrac{1}{T}\sum_t (c_t - \hat c_t)^2}$.
- Rolloff RMSE (Hz): $\sqrt{\tfrac{1}{T}\sum_t (r_t - \hat r_t)^2}$.
- Multi-resolution STFT (MR-STFT): Mean over several STFT configurations of $0.5\,(\text{log-mag L1} + \text{spectral convergence})$.
- Log-mel L1: $\|\log M - \log \hat M\|_1$ between log-mel spectrograms.
- Combined mel_mrstft: $0.5\,(\text{Log-mel L1} + \text{MR-STFT})$.

Composite (if shown) is the unweighted mean of normalized metrics present.

## Known limitations

- Current FM signal path is a fixed sine 4-osc chain; not all Operator routings are modeled.
- Preset export assumes specific Ableton Operator algorithms (FM 0, additive 10).
- Target partial placement uses a simple kernel on the FFT grid; no phase modeling.
- MFCC distance relies on `librosa` defaults; tuning may be task-dependent.

## Reproducibility

- Most routines accept a `seed` (default 42) for reproducible runs.
- Generated files include timestamps and method/metric suffixes for traceability.

## References

- Risset, J.-C., & Wessel, D. (1999). Exploration of timbre by analysis and synthesis. In D. Deutsch (Ed.), The psychology of music (pp. 113-169). Academic Press.
- SciPy optimization (`differential_evolution`, `dual_annealing`, `basinhopping`): [scipy.org](https://scipy.org)
- Librosa features (MFCC): [librosa.org](https://librosa.org)

## Contributing

Contributions, issues, and feature requests are welcome! Please check the issues page and open a discussion before major changes. For small fixes or documentation improvements, feel free to submit a pull request.

- Issues: https://github.com/egorpol/FFTimbre/issues
- Pull requests: https://github.com/egorpol/FFTimbre/pulls

## How to Cite

This work was presented at InMusic'24. If you use this toolkit in your research, please cite the upcoming paper in the conference proceedings (Routledge). A pre-print or final citation will be provided here when available.

Until then, you can cite the project as:

```bibtex
@misc{FFTimbre,
  title        = {FFTimbre: Spectral Matching via FM and Additive Synthesis},
  year         = {2025},
  howpublished = {\url{https://github.com/egorpol/FFTimbre}},
  note         = {Accessed: 2025-09-07}
}
```

## License

MIT - see `LICENSE`.
