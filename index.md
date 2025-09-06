---
layout: default
title: FFTimbre Showcase
---

<link rel="stylesheet" href="{{ '/assets/style.css' | relative_url }}">

# FFTimbre Showcase

This page hosts audio samples and plots produced by the notebooks in this repo. For background see:
- [General concept](https://github.com/egorpol/FFTimbre#general-concept) · [Metrics & optimizers](https://github.com/egorpol/FFTimbre#metrics-and-optimizers) · [Repository structure](https://github.com/egorpol/FFTimbre#repository-structure)

## Legend
- [FFT frame (bowed cello)](#cello-fft-frame)
- [Target spectra](#target-spectra)
- [Optimized FM — DE + cosine](#optimized-fm-with-de-cosine)
- [Optimized FM — DE + euclidean](#optimized-fm-with-de-euclidean)
- [Optimized FM — DE + itakura_saito](#optimized-fm-with-de-itakura-saito)
- [Optimized FM — DE + kl](#optimized-fm-with-de-kl)
- [Optimized FM — DE + manhattan](#optimized-fm-with-de-manhattan)
- [Optimized FM — DE + mfcc](#optimized-fm-with-de-mfcc)
- [Optimized FM — DE + pearson](#optimized-fm-with-de-pearson)
- [Optimized FM — DE + spectral_convergence](#optimized-fm-with-de-spectral-convergence)

<a id="cello-fft-frame"></a>
## FFT Frame based on bowed cello sound

Source: [`tsv/cello_single.tsv`]({{ '/tsv/cello_single.tsv' | relative_url }})

{% include tsv_table.html 
   src="/tsv/cello_single.tsv"
   has_header=true
   max_rows=50
   caption="First 50 rows of cello_single.tsv"
%}

{% include sample.html 
   title="Target spectra"
   description="Reference spectrogram and waveform for the bowed cello FFT frame."
   audio="/rendered_audio/additive_from_cello_single_2.0s_20250906-215542.wav"
   plot="/rendered_plots/additive_from_cello_single_2.0s_20250906-215542_spectrum.png|/rendered_plots/additive_from_cello_single_2.0s_20250906-215542_time.png"
   captions="Spectrogram|Waveform"
%}

Results below are from a batch run (`maxiter=500`) across metrics using Differential Evolution.

{% include sample.html 
   title="Optimized FM with DE + cosine"
   description="FM resynthesis optimized with Differential Evolution using cosine similarity."
   audio="/rendered_audio/optimized_output_fm_cello_single_de_cosine_20250904-003735.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_de_cosine_20250904-003735_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_de_cosine_20250904-003735_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DE + euclidean"
   description="FM resynthesis optimized with Differential Evolution using Euclidean distance."
   audio="/rendered_audio/optimized_output_fm_cello_single_de_euclidean_20250904-004049.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_de_euclidean_20250904-004049_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_de_euclidean_20250904-004049_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DE + itakura_saito"
   description="FM resynthesis optimized with Differential Evolution using Itakura–Saito divergence."
   audio="/rendered_audio/optimized_output_fm_cello_single_de_itakura_saito_20250904-003136.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_de_itakura_saito_20250904-003136_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_de_itakura_saito_20250904-003136_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DE + kl"
   description="FM resynthesis optimized with Differential Evolution using KL divergence."
   audio="/rendered_audio/optimized_output_fm_cello_single_de_kl_20250904-004429.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_de_kl_20250904-004429_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_de_kl_20250904-004429_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DE + manhattan"
   description="FM resynthesis optimized with Differential Evolution using Manhattan distance."
   audio="/rendered_audio/optimized_output_fm_cello_single_de_manhattan_20250904-004211.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_de_manhattan_20250904-004211_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_de_manhattan_20250904-004211_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DE + mfcc"
   description="FM resynthesis optimized with Differential Evolution using MFCC distance."
   audio="/rendered_audio/optimized_output_fm_cello_single_de_mfcc_20250904-002651.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_de_mfcc_20250904-002651_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_de_mfcc_20250904-002651_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DE + pearson"
   description="FM resynthesis optimized with Differential Evolution using Pearson correlation."
   audio="/rendered_audio/optimized_output_fm_cello_single_de_pearson_20250904-002050.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_de_pearson_20250904-002050_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_de_pearson_20250904-002050_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DE + spectral_convergence"
   description="FM resynthesis optimized with Differential Evolution using spectral convergence."
   audio="/rendered_audio/optimized_output_fm_cello_single_de_spectral_convergence_20250904-003453.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_de_spectral_convergence_20250904-003453_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_de_spectral_convergence_20250904-003453_time.png"
   captions="Spectrogram|Waveform"
%}
