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
- [Optimized FM — DA + cosine](#optimized-fm-with-da-cosine)
- [Optimized FM — DA + euclidean](#optimized-fm-with-da-euclidean)
- [Optimized FM — DA + itakura_saito](#optimized-fm-with-da-itakura-saito)
- [Optimized FM — DA + kl](#optimized-fm-with-da-kl)
- [Optimized FM — DA + manhattan](#optimized-fm-with-da-manhattan)
- [Optimized FM — DA + mfcc](#optimized-fm-with-da-mfcc)
- [Optimized FM — DA + pearson](#optimized-fm-with-da-pearson)
- [Optimized FM — DA + spectral_convergence](#optimized-fm-with-da-spectral-convergence)
 - [Optimized FM — BH + cosine](#optimized-fm-with-bh-cosine)
 - [Optimized FM — BH + euclidean](#optimized-fm-with-bh-euclidean)
 - [Optimized FM — BH + itakura_saito](#optimized-fm-with-bh-itakura-saito)
 - [Optimized FM — BH + kl](#optimized-fm-with-bh-kl)
 - [Optimized FM — BH + manhattan](#optimized-fm-with-bh-manhattan)
 - [Optimized FM — BH + mfcc](#optimized-fm-with-bh-mfcc)
 - [Optimized FM — BH + pearson](#optimized-fm-with-bh-pearson)
 - [Optimized FM — BH + spectral_convergence](#optimized-fm-with-bh-spectral-convergence)

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

Results below are from a batch run (`maxiter=500`) across metrics using Dual Annealing.

{% include sample.html 
   title="Optimized FM with DA + cosine"
   description="FM resynthesis optimized with Dual Annealing using cosine similarity."
   audio="/rendered_audio/optimized_output_fm_cello_single_da_cosine_20250904-003801.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_da_cosine_20250904-003801_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_da_cosine_20250904-003801_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DA + euclidean"
   description="FM resynthesis optimized with Dual Annealing using Euclidean distance."
   audio="/rendered_audio/optimized_output_fm_cello_single_da_euclidean_20250904-004117.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_da_euclidean_20250904-004117_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_da_euclidean_20250904-004117_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DA + itakura_saito"
   description="FM resynthesis optimized with Dual Annealing using Itakura–Saito divergence."
   audio="/rendered_audio/optimized_output_fm_cello_single_da_itakura_saito_20250904-003210.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_da_itakura_saito_20250904-003210_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_da_itakura_saito_20250904-003210_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DA + kl"
   description="FM resynthesis optimized with Dual Annealing using KL divergence."
   audio="/rendered_audio/optimized_output_fm_cello_single_da_kl_20250904-004455.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_da_kl_20250904-004455_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_da_kl_20250904-004455_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DA + manhattan"
   description="FM resynthesis optimized with Dual Annealing using Manhattan distance."
   audio="/rendered_audio/optimized_output_fm_cello_single_da_manhattan_20250904-004233.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_da_manhattan_20250904-004233_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_da_manhattan_20250904-004233_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DA + mfcc"
   description="FM resynthesis optimized with Dual Annealing using MFCC distance."
   audio="/rendered_audio/optimized_output_fm_cello_single_da_mfcc_20250904-002818.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_da_mfcc_20250904-002818_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_da_mfcc_20250904-002818_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DA + pearson"
   description="FM resynthesis optimized with Dual Annealing using Pearson correlation."
   audio="/rendered_audio/optimized_output_fm_cello_single_da_pearson_20250904-002136.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_da_pearson_20250904-002136_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_da_pearson_20250904-002136_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DA + spectral_convergence"
   description="FM resynthesis optimized with Dual Annealing using spectral convergence."
   audio="/rendered_audio/optimized_output_fm_cello_single_da_spectral_convergence_20250904-003515.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_da_spectral_convergence_20250904-003515_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_da_spectral_convergence_20250904-003515_time.png"
   captions="Spectrogram|Waveform"
%}

Results below are from a batch run (`maxiter=500`) across metrics using Basin Hopping.

{% include sample.html 
   title="Optimized FM with BH + cosine"
   description="FM resynthesis optimized with Basin Hopping using cosine similarity."
   audio="/rendered_audio/optimized_output_fm_cello_single_bh_cosine_20250904-003840.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_bh_cosine_20250904-003840_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_bh_cosine_20250904-003840_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with BH + euclidean"
   description="FM resynthesis optimized with Basin Hopping using Euclidean distance."
   audio="/rendered_audio/optimized_output_fm_cello_single_bh_euclidean_20250904-004138.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_bh_euclidean_20250904-004138_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_bh_euclidean_20250904-004138_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with BH + itakura_saito"
   description="FM resynthesis optimized with Basin Hopping using Itakura–Saito divergence."
   audio="/rendered_audio/optimized_output_fm_cello_single_bh_itakura_saito_20250904-003244.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_bh_itakura_saito_20250904-003244_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_bh_itakura_saito_20250904-003244_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with BH + kl"
   description="FM resynthesis optimized with Basin Hopping using KL divergence."
   audio="/rendered_audio/optimized_output_fm_cello_single_bh_kl_20250904-004529.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_bh_kl_20250904-004529_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_bh_kl_20250904-004529_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with BH + manhattan"
   description="FM resynthesis optimized with Basin Hopping using Manhattan distance."
   audio="/rendered_audio/optimized_output_fm_cello_single_bh_manhattan_20250904-004305.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_bh_manhattan_20250904-004305_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_bh_manhattan_20250904-004305_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with BH + mfcc"
   description="FM resynthesis optimized with Basin Hopping using MFCC distance."
   audio="/rendered_audio/optimized_output_fm_cello_single_bh_mfcc_20250904-003015.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_bh_mfcc_20250904-003015_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_bh_mfcc_20250904-003015_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with BH + pearson"
   description="FM resynthesis optimized with Basin Hopping using Pearson correlation."
   audio="/rendered_audio/optimized_output_fm_cello_single_bh_pearson_20250904-002243.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_bh_pearson_20250904-002243_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_bh_pearson_20250904-002243_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with BH + spectral_convergence"
   description="FM resynthesis optimized with Basin Hopping using spectral convergence."
   audio="/rendered_audio/optimized_output_fm_cello_single_bh_spectral_convergence_20250904-003533.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_bh_spectral_convergence_20250904-003533_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_bh_spectral_convergence_20250904-003533_time.png"
   captions="Spectrogram|Waveform"
%}
