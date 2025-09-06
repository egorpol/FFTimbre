# FFTimbre

FFTimbre is a small toolkit of notebooks and helpers for spectral matching via FM and additive synthesis. It started as an exploratory idea inside [audiospylt](https://github.com/egorpol/audiospylt) and has since grown into its own project.

This setup was introduced in a talk at the InMusic'24 conference (Oslo). Slides are available in `inmusic24_slides/`. A paper version is planned for the InMusic'24 proceedings (Routledge).

## Overview

The core question: how can we reproduce a target spectrum with a constrained oscillator setup? The initial motivation was to automatically derive settings for a Doepfer A‑100 modular configuration (mostly for fun), but it has evolved into a general tool for resynthesizing spectra, with export to Ableton Operator in mind.

The approach is inspired by ideas from Risset & Wessel (1999), where a human judges the “goodness of fit” by comparing synthesized and original sounds. Here, we explore automated evaluation via spectral metrics and global optimizers.

![Risset/Wessel concept](risset.png)

## General concept

- We take a table of frequency/amplitude pairs (e.g., from [audiospylt](https://github.com/egorpol/audiospylt)) as the target.
- We compare target and synthesized spectra using multiple similarity metrics:
  - Itakura–Saito, Spectral Convergence, Cosine, Euclidean, Manhattan, KL, Pearson, MFCC distance.
- We optimize synthesis parameters using multiple global optimizers:
  - Differential Evolution (DE), Dual Annealing (DA), Basin Hopping (BH).
A graphical overview:

![Workflow schema](schema.png)

After optimization we get a TSV with oscillator values. We can synthesize it in Python right away or export it as an Ableton Operator preset.
We target two Operator setups: FM (algorithm 0 — the leftmost in the Operator GUI) and additive (algorithm 10 — the rightmost).
To go from a TSV of optimized oscillator settings to the Operator preset, we adjust value formatting and write the preset data into a compressed XML file.

## Repository Structure

- `inmusic24_slides/` — conference slides (PDF/PPTX)
- `py_scripts/`
  - `fm_synth_opt.py` — objective, synthesis, optimization runners, plotting, I/O helpers
  - `optimization_workflow.py` — batch job utilities and consistent file naming
  - `objective_functions.py` — legacy/experimental objective prototypes
  - `generate_wave_file.py` — WAV rendering at various sample rates/bit depths
  - `interactive_controls.py` — reusable ipywidgets UI for notebook workflows
  - `waveform_generators.py` — basic waveform functions (sine/square/triangle/saw/noise)
- `tsv/` — example target partials and saved final oscillator values (e.g., `cello_single.tsv`, `noanoa_single.tsv`, `final_values_fm_*.tsv`)
- `xml_presets/` — example Operator preset files (`.adv` Ableton presets and `.xml` sources)
- `rendered_audio/` — generated WAV files
- `rendered_gifs/` — optimizer animations
- `rendered_plots/` — saved PNG plots from runs

- `risset.png`, `schema.png` — figures used in the README

- Notebooks
  - `distances_demo.ipynb` — overview of objective functions on a simple spectrum
  - `optimization_gif.ipynb` — preview of the optimization algorithm workflow
  - `parameter_operator_fm.ipynb`, `parameter_operator_additive.ipynb` — parameter mapping
  - `preset_editor_operator_fm.ipynb`, `preset_editor_operator_additive.ipynb` — preset export/editing
  - `spectral_ml_additive3.ipynb` — additive synthesis experiments
  - `spectral_ml_fm3.ipynb`, `spectral_ml_fm3_interactive.ipynb` — earlier/interactive FM experiments
  - `spectral_ml_fm4.ipynb` — guided single‑run workflow (optimize → preview → save)
  - `spectral_ml_fm4_batch.ipynb` — batch runner for metrics/optimizers with consistent naming
  - `render_from_tsv_fm.ipynb` — render audio from saved TSV with optimized FM values
  - `render_from_tsv_table.ipynb`

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

- **Metrics**: Itakura–Saito (`itakura_saito`), Spectral Convergence (`spectral_convergence`), Cosine (`cosine`), Euclidean (`euclidean`), Manhattan (`manhattan`), KL (`kl`), Pearson (`pearson`), MFCC (`mfcc`).
- **Optimizers**: Differential Evolution (DE), Dual Annealing (DA), Basin Hopping (BH).

## Known limitations

- Current FM signal path is a fixed 4‑osc chain; not all Operator routings are modeled.
- Preset export assumes specific Ableton Operator algorithms (FM 0, additive 10).
- Target partial placement uses a simple kernel on the FFT grid; no phase modeling.
- MFCC distance relies on `librosa` defaults; tuning may be task‑dependent.

## Reproducibility

- Most routines accept a `seed` (default 42) for reproducible runs.
- Generated files include timestamps and method/metric suffixes for traceability.

## References

- Risset, J.-C., & Wessel, D. (1999). Exploration of timbre by analysis and synthesis. In D. Deutsch (Ed.), The psychology of music (pp. 113–169). Academic Press.
- SciPy optimization (`differential_evolution`, `dual_annealing`, `basinhopping`): [scipy.org](https://scipy.org)
- Librosa features (MFCC): [librosa.org](https://librosa.org)

## License

MIT — see `LICENSE`.

