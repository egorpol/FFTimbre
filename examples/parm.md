---
layout: default
title: Parmegiani — "Géologie Sonore" Onset (single frame)
---

<link rel="stylesheet" href="{{ '/assets/style.css' | relative_url }}">

# Parmegiani — "Géologie Sonore" Onset (single frame)

Iconic synth-like sound sampled from the beginning of Parmegiani’s "Géologie Sonore" (De Natura Sonorum, Première série).

- Source TSV: [`tsv/parm.tsv`]({{ '/tsv/parm.tsv' | relative_url }})
- Back to overview: [FFTimbre Showcase]({{ '/' | relative_url }})

## Legend
- [FFT frame (Parmegiani onset)](#parm-fft-frame)
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
 - [Optimized Additive — DE + cosine](#optimized-additive-with-de-cosine)
 - [Optimized Additive — DE + euclidean](#optimized-additive-with-de-euclidean)
 - [Optimized Additive — DE + itakura_saito](#optimized-additive-with-de-itakura-saito)
 - [Optimized Additive — DE + kl](#optimized-additive-with-de-kl)
 - [Optimized Additive — DE + manhattan](#optimized-additive-with-de-manhattan)
 - [Optimized Additive — DE + mfcc](#optimized-additive-with-de-mfcc)
 - [Optimized Additive — DE + pearson](#optimized-additive-with-de-pearson)
 - [Optimized Additive — DE + spectral_convergence](#optimized-additive-with-de-spectral-convergence)
 - [Optimized Additive — DA + cosine](#optimized-additive-with-da-cosine)
 - [Optimized Additive — DA + euclidean](#optimized-additive-with-da-euclidean)
 - [Optimized Additive — DA + itakura_saito](#optimized-additive-with-da-itakura-saito)
 - [Optimized Additive — DA + kl](#optimized-additive-with-da-kl)
 - [Optimized Additive — DA + manhattan](#optimized-additive-with-da-manhattan)
 - [Optimized Additive — DA + mfcc](#optimized-additive-with-da-mfcc)
 - [Optimized Additive — DA + pearson](#optimized-additive-with-da-pearson)
 - [Optimized Additive — DA + spectral_convergence](#optimized-additive-with-da-spectral-convergence)
 - [Optimized Additive — BH + cosine](#optimized-additive-with-bh-cosine)
 - [Optimized Additive — BH + euclidean](#optimized-additive-with-bh-euclidean)
 - [Optimized Additive — BH + itakura_saito](#optimized-additive-with-bh-itakura-saito)
 - [Optimized Additive — BH + kl](#optimized-additive-with-bh-kl)
 - [Optimized Additive — BH + manhattan](#optimized-additive-with-bh-manhattan)
 - [Optimized Additive — BH + mfcc](#optimized-additive-with-bh-mfcc)
 - [Optimized Additive — BH + pearson](#optimized-additive-with-bh-pearson)
 - [Optimized Additive — BH + spectral_convergence](#optimized-additive-with-bh-spectral-convergence)

<a id="parm-fft-frame"></a>
## FFT Frame based on Parmegiani onset

{% include tsv_table.html 
   src="/tsv/parm.tsv"
   has_header=true
   max_rows=50
   caption="First 50 rows of parm.tsv"
%}

{% include sample.html 
   title="Target spectra"
   description="Reference spectrogram and waveform for the Parmegiani onset FFT frame."
   audio="/rendered_audio/additive_from_parm_2.0s_20250908-142704.wav"
   plot="/rendered_plots/additive_from_parm_2.0s_20250908-142704_spectrum.png|/rendered_plots/additive_from_parm_2.0s_20250908-142704_time.png"
   captions="Spectrogram|Waveform"
%}

Results below are from a batch run (`maxiter=500`) across metrics using Differential Evolution.

{% include sample.html 
   title="Optimized FM with DE + cosine"
   description="FM resynthesis optimized with Differential Evolution using cosine similarity."
   audio="/rendered_audio/optimized_output_fm_parm_de_cosine_20250908-150414.wav"
   plot="/rendered_plots/optimized_output_fm_parm_de_cosine_20250908-150414_spectrum.png|/rendered_plots/optimized_output_fm_parm_de_cosine_20250908-150414_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DE + euclidean"
   description="FM resynthesis optimized with Differential Evolution using Euclidean distance."
   audio="/rendered_audio/optimized_output_fm_parm_de_euclidean_20250908-150743.wav"
   plot="/rendered_plots/optimized_output_fm_parm_de_euclidean_20250908-150743_spectrum.png|/rendered_plots/optimized_output_fm_parm_de_euclidean_20250908-150743_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DE + itakura_saito"
   description="FM resynthesis optimized with Differential Evolution using Itakura–Saito divergence."
   audio="/rendered_audio/optimized_output_fm_parm_de_itakura_saito_20250908-145757.wav"
   plot="/rendered_plots/optimized_output_fm_parm_de_itakura_saito_20250908-145757_spectrum.png|/rendered_plots/optimized_output_fm_parm_de_itakura_saito_20250908-145757_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DE + kl"
   description="FM resynthesis optimized with Differential Evolution using KL divergence."
   audio="/rendered_audio/optimized_output_fm_parm_de_kl_20250908-151130.wav"
   plot="/rendered_plots/optimized_output_fm_parm_de_kl_20250908-151130_spectrum.png|/rendered_plots/optimized_output_fm_parm_de_kl_20250908-151130_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DE + manhattan"
   description="FM resynthesis optimized with Differential Evolution using Manhattan distance."
   audio="/rendered_audio/optimized_output_fm_parm_de_manhattan_20250908-150902.wav"
   plot="/rendered_plots/optimized_output_fm_parm_de_manhattan_20250908-150902_spectrum.png|/rendered_plots/optimized_output_fm_parm_de_manhattan_20250908-150902_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DE + mfcc"
   description="FM resynthesis optimized with Differential Evolution using MFCC distance."
   audio="/rendered_audio/optimized_output_fm_parm_de_mfcc_20250908-145245.wav"
   plot="/rendered_plots/optimized_output_fm_parm_de_mfcc_20250908-145245_spectrum.png|/rendered_plots/optimized_output_fm_parm_de_mfcc_20250908-145245_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DE + pearson"
   description="FM resynthesis optimized with Differential Evolution using Pearson correlation."
   audio="/rendered_audio/optimized_output_fm_parm_de_pearson_20250908-144626.wav"
   plot="/rendered_plots/optimized_output_fm_parm_de_pearson_20250908-144626_spectrum.png|/rendered_plots/optimized_output_fm_parm_de_pearson_20250908-144626_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DE + spectral_convergence"
   description="FM resynthesis optimized with Differential Evolution using spectral convergence."
   audio="/rendered_audio/optimized_output_fm_parm_de_spectral_convergence_20250908-150114.wav"
   plot="/rendered_plots/optimized_output_fm_parm_de_spectral_convergence_20250908-150114_spectrum.png|/rendered_plots/optimized_output_fm_parm_de_spectral_convergence_20250908-150114_time.png"
   captions="Spectrogram|Waveform"
%}

Results below are from a batch run (`maxiter=500`) across metrics using Dual Annealing.

{% include sample.html 
   title="Optimized FM with DA + cosine"
   description="FM resynthesis optimized with Dual Annealing using cosine similarity."
   audio="/rendered_audio/optimized_output_fm_parm_da_cosine_20250908-150447.wav"
   plot="/rendered_plots/optimized_output_fm_parm_da_cosine_20250908-150447_spectrum.png|/rendered_plots/optimized_output_fm_parm_da_cosine_20250908-150447_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DA + euclidean"
   description="FM resynthesis optimized with Dual Annealing using Euclidean distance."
   audio="/rendered_audio/optimized_output_fm_parm_da_euclidean_20250908-150812.wav"
   plot="/rendered_plots/optimized_output_fm_parm_da_euclidean_20250908-150812_spectrum.png|/rendered_plots/optimized_output_fm_parm_da_euclidean_20250908-150812_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DA + itakura_saito"
   description="FM resynthesis optimized with Dual Annealing using Itakura–Saito divergence."
   audio="/rendered_audio/optimized_output_fm_parm_da_itakura_saito_20250908-145822.wav"
   plot="/rendered_plots/optimized_output_fm_parm_da_itakura_saito_20250908-145822_spectrum.png|/rendered_plots/optimized_output_fm_parm_da_itakura_saito_20250908-145822_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DA + kl"
   description="FM resynthesis optimized with Dual Annealing using KL divergence."
   audio="/rendered_audio/optimized_output_fm_parm_da_kl_20250908-151159.wav"
   plot="/rendered_plots/optimized_output_fm_parm_da_kl_20250908-151159_spectrum.png|/rendered_plots/optimized_output_fm_parm_da_kl_20250908-151159_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DA + manhattan"
   description="FM resynthesis optimized with Dual Annealing using Manhattan distance."
   audio="/rendered_audio/optimized_output_fm_parm_da_manhattan_20250908-150924.wav"
   plot="/rendered_plots/optimized_output_fm_parm_da_manhattan_20250908-150924_spectrum.png|/rendered_plots/optimized_output_fm_parm_da_manhattan_20250908-150924_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DA + mfcc"
   description="FM resynthesis optimized with Dual Annealing using MFCC distance."
   audio="/rendered_audio/optimized_output_fm_parm_da_mfcc_20250908-145411.wav"
   plot="/rendered_plots/optimized_output_fm_parm_da_mfcc_20250908-145411_spectrum.png|/rendered_plots/optimized_output_fm_parm_da_mfcc_20250908-145411_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DA + pearson"
   description="FM resynthesis optimized with Dual Annealing using Pearson correlation."
   audio="/rendered_audio/optimized_output_fm_parm_da_pearson_20250908-144704.wav"
   plot="/rendered_plots/optimized_output_fm_parm_da_pearson_20250908-144704_spectrum.png|/rendered_plots/optimized_output_fm_parm_da_pearson_20250908-144704_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DA + spectral_convergence"
   description="FM resynthesis optimized with Dual Annealing using spectral convergence."
   audio="/rendered_audio/optimized_output_fm_parm_da_spectral_convergence_20250908-150143.wav"
   plot="/rendered_plots/optimized_output_fm_parm_da_spectral_convergence_20250908-150143_spectrum.png|/rendered_plots/optimized_output_fm_parm_da_spectral_convergence_20250908-150143_time.png"
   captions="Spectrogram|Waveform"
%}

Results below are from a batch run (`maxiter=500`) across metrics using Basin Hopping.

{% include sample.html 
   title="Optimized FM with BH + cosine"
   description="FM resynthesis optimized with Basin Hopping using cosine similarity."
   audio="/rendered_audio/optimized_output_fm_parm_bh_cosine_20250908-150534.wav"
   plot="/rendered_plots/optimized_output_fm_parm_bh_cosine_20250908-150534_spectrum.png|/rendered_plots/optimized_output_fm_parm_bh_cosine_20250908-150534_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with BH + euclidean"
   description="FM resynthesis optimized with Basin Hopping using Euclidean distance."
   audio="/rendered_audio/optimized_output_fm_parm_bh_euclidean_20250908-150834.wav"
   plot="/rendered_plots/optimized_output_fm_parm_bh_euclidean_20250908-150834_spectrum.png|/rendered_plots/optimized_output_fm_parm_bh_euclidean_20250908-150834_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with BH + itakura_saito"
   description="FM resynthesis optimized with Basin Hopping using Itakura–Saito divergence."
   audio="/rendered_audio/optimized_output_fm_parm_bh_itakura_saito_20250908-145858.wav"
   plot="/rendered_plots/optimized_output_fm_parm_bh_itakura_saito_20250908-145858_spectrum.png|/rendered_plots/optimized_output_fm_parm_bh_itakura_saito_20250908-145858_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with BH + kl"
   description="FM resynthesis optimized with Basin Hopping using KL divergence."
   audio="/rendered_audio/optimized_output_fm_parm_bh_kl_20250908-151243.wav"
   plot="/rendered_plots/optimized_output_fm_parm_bh_kl_20250908-151243_spectrum.png|/rendered_plots/optimized_output_fm_parm_bh_kl_20250908-151243_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with BH + manhattan"
   description="FM resynthesis optimized with Basin Hopping using Manhattan distance."
   audio="/rendered_audio/optimized_output_fm_parm_bh_manhattan_20250908-150959.wav"
   plot="/rendered_plots/optimized_output_fm_parm_bh_manhattan_20250908-150959_spectrum.png|/rendered_plots/optimized_output_fm_parm_bh_manhattan_20250908-150959_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with BH + mfcc"
   description="FM resynthesis optimized with Basin Hopping using MFCC distance."
   audio="/rendered_audio/optimized_output_fm_parm_bh_mfcc_20250908-145634.wav"
   plot="/rendered_plots/optimized_output_fm_parm_bh_mfcc_20250908-145634_spectrum.png|/rendered_plots/optimized_output_fm_parm_bh_mfcc_20250908-145634_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with BH + pearson"
   description="FM resynthesis optimized with Basin Hopping using Pearson correlation."
   audio="/rendered_audio/optimized_output_fm_parm_bh_pearson_20250908-144818.wav"
   plot="/rendered_plots/optimized_output_fm_parm_bh_pearson_20250908-144818_spectrum.png|/rendered_plots/optimized_output_fm_parm_bh_pearson_20250908-144818_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with BH + spectral_convergence"
   description="FM resynthesis optimized with Basin Hopping using spectral convergence."
   audio="/rendered_audio/optimized_output_fm_parm_bh_spectral_convergence_20250908-150206.wav"
   plot="/rendered_plots/optimized_output_fm_parm_bh_spectral_convergence_20250908-150206_spectrum.png|/rendered_plots/optimized_output_fm_parm_bh_spectral_convergence_20250908-150206_time.png"
   captions="Spectrogram|Waveform"
%}

Results below are from a batch run (maxiter=500) across metrics using Differential Evolution.

{% include sample.html 
   title="Optimized Additive with DE + cosine"
   description="Additive resynthesis optimized with Differential Evolution using cosine similarity."
   audio="/rendered_audio/optimized_output_additive_parm_de_cosine_20250911-145449.wav"
   plot="/rendered_plots/optimized_output_additive_parm_de_cosine_20250911-145449_spectrum.png|/rendered_plots/optimized_output_additive_parm_de_cosine_20250911-145449_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with DE + euclidean"
   description="Additive resynthesis optimized with Differential Evolution using Euclidean distance."
   audio="/rendered_audio/optimized_output_additive_parm_de_euclidean_20250911-150843.wav"
   plot="/rendered_plots/optimized_output_additive_parm_de_euclidean_20250911-150843_spectrum.png|/rendered_plots/optimized_output_additive_parm_de_euclidean_20250911-150843_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with DE + itakura_saito"
   description="Additive resynthesis optimized with Differential Evolution using Itakura–Saito divergence."
   audio="/rendered_audio/optimized_output_additive_parm_de_itakura_saito_20250911-142345.wav"
   plot="/rendered_plots/optimized_output_additive_parm_de_itakura_saito_20250911-142345_spectrum.png|/rendered_plots/optimized_output_additive_parm_de_itakura_saito_20250911-142345_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with DE + kl"
   description="Additive resynthesis optimized with Differential Evolution using KL divergence."
   audio="/rendered_audio/optimized_output_additive_parm_de_kl_20250911-153657.wav"
   plot="/rendered_plots/optimized_output_additive_parm_de_kl_20250911-153657_spectrum.png|/rendered_plots/optimized_output_additive_parm_de_kl_20250911-153657_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with DE + manhattan"
   description="Additive resynthesis optimized with Differential Evolution using Manhattan distance."
   audio="/rendered_audio/optimized_output_additive_parm_de_manhattan_20250911-152815.wav"
   plot="/rendered_plots/optimized_output_additive_parm_de_manhattan_20250911-152815_spectrum.png|/rendered_plots/optimized_output_additive_parm_de_manhattan_20250911-152815_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with DE + mfcc"
   description="Additive resynthesis optimized with Differential Evolution using MFCC distance."
   audio="/rendered_audio/optimized_output_additive_parm_de_mfcc_20250911-140638.wav"
   plot="/rendered_plots/optimized_output_additive_parm_de_mfcc_20250911-140638_spectrum.png|/rendered_plots/optimized_output_additive_parm_de_mfcc_20250911-140638_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with DE + pearson"
   description="Additive resynthesis optimized with Differential Evolution using Pearson correlation."
   audio="/rendered_audio/optimized_output_additive_parm_de_pearson_20250911-130943.wav"
   plot="/rendered_plots/optimized_output_additive_parm_de_pearson_20250911-130943_spectrum.png|/rendered_plots/optimized_output_additive_parm_de_pearson_20250911-130943_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with DE + spectral_convergence"
   description="Additive resynthesis optimized with Differential Evolution using spectral convergence."
   audio="/rendered_audio/optimized_output_additive_parm_de_spectral_convergence_20250911-143519.wav"
   plot="/rendered_plots/optimized_output_additive_parm_de_spectral_convergence_20250911-143519_spectrum.png|/rendered_plots/optimized_output_additive_parm_de_spectral_convergence_20250911-143519_time.png"
   captions="Spectrogram|Waveform"
%}

Results below are from a batch run (maxiter=500) across metrics using Dual Annealing.

{% include sample.html 
   title="Optimized Additive with DA + cosine"
   description="Additive resynthesis optimized with Dual Annealing using cosine similarity."
   audio="/rendered_audio/optimized_output_additive_parm_da_cosine_20250911-145627.wav"
   plot="/rendered_plots/optimized_output_additive_parm_da_cosine_20250911-145627_spectrum.png|/rendered_plots/optimized_output_additive_parm_da_cosine_20250911-145627_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with DA + euclidean"
   description="Additive resynthesis optimized with Dual Annealing using Euclidean distance."
   audio="/rendered_audio/optimized_output_additive_parm_da_euclidean_20250911-151046.wav"
   plot="/rendered_plots/optimized_output_additive_parm_da_euclidean_20250911-151046_spectrum.png|/rendered_plots/optimized_output_additive_parm_da_euclidean_20250911-151046_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with DA + itakura_saito"
   description="Additive resynthesis optimized with Dual Annealing using Itakura–Saito divergence."
   audio="/rendered_audio/optimized_output_additive_parm_da_itakura_saito_20250911-142845.wav"
   plot="/rendered_plots/optimized_output_additive_parm_da_itakura_saito_20250911-142845_spectrum.png|/rendered_plots/optimized_output_additive_parm_da_itakura_saito_20250911-142845_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with DA + kl"
   description="Additive resynthesis optimized with Dual Annealing using KL divergence."
   audio="/rendered_audio/optimized_output_additive_parm_da_kl_20250911-154143.wav"
   plot="/rendered_plots/optimized_output_additive_parm_da_kl_20250911-154143_spectrum.png|/rendered_plots/optimized_output_additive_parm_da_kl_20250911-154143_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with DA + manhattan"
   description="Additive resynthesis optimized with Dual Annealing using Manhattan distance."
   audio="/rendered_audio/optimized_output_additive_parm_da_manhattan_20250911-153028.wav"
   plot="/rendered_plots/optimized_output_additive_parm_da_manhattan_20250911-153028_spectrum.png|/rendered_plots/optimized_output_additive_parm_da_manhattan_20250911-153028_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with DA + mfcc"
   description="Additive resynthesis optimized with Dual Annealing using MFCC distance."
   audio="/rendered_audio/optimized_output_additive_parm_da_mfcc_20250911-141720.wav"
   plot="/rendered_plots/optimized_output_additive_parm_da_mfcc_20250911-141720_spectrum.png|/rendered_plots/optimized_output_additive_parm_da_mfcc_20250911-141720_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with DA + pearson"
   description="Additive resynthesis optimized with Dual Annealing using Pearson correlation."
   audio="/rendered_audio/optimized_output_additive_parm_da_pearson_20250911-131453.wav"
   plot="/rendered_plots/optimized_output_additive_parm_da_pearson_20250911-131453_spectrum.png|/rendered_plots/optimized_output_additive_parm_da_pearson_20250911-131453_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with DA + spectral_convergence"
   description="Additive resynthesis optimized with Dual Annealing using spectral convergence."
   audio="/rendered_audio/optimized_output_additive_parm_da_spectral_convergence_20250911-143712.wav"
   plot="/rendered_plots/optimized_output_additive_parm_da_spectral_convergence_20250911-143712_spectrum.png|/rendered_plots/optimized_output_additive_parm_da_spectral_convergence_20250911-143712_time.png"
   captions="Spectrogram|Waveform"
%}

Results below are from a batch run (maxiter=500) across metrics using Basin Hopping.

{% include sample.html 
   title="Optimized Additive with BH + cosine"
   description="Additive resynthesis optimized with Basin Hopping using cosine similarity."
   audio="/rendered_audio/optimized_output_additive_parm_bh_cosine_20250911-150507.wav"
   plot="/rendered_plots/optimized_output_additive_parm_bh_cosine_20250911-150507_spectrum.png|/rendered_plots/optimized_output_additive_parm_bh_cosine_20250911-150507_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with BH + euclidean"
   description="Additive resynthesis optimized with Basin Hopping using Euclidean distance."
   audio="/rendered_audio/optimized_output_additive_parm_bh_euclidean_20250911-152547.wav"
   plot="/rendered_plots/optimized_output_additive_parm_bh_euclidean_20250911-152547_spectrum.png|/rendered_plots/optimized_output_additive_parm_bh_euclidean_20250911-152547_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with BH + itakura_saito"
   description="Additive resynthesis optimized with Basin Hopping using Itakura–Saito divergence."
   audio="/rendered_audio/optimized_output_additive_parm_bh_itakura_saito_20250911-143206.wav"
   plot="/rendered_plots/optimized_output_additive_parm_bh_itakura_saito_20250911-143206_spectrum.png|/rendered_plots/optimized_output_additive_parm_bh_itakura_saito_20250911-143206_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with BH + kl"
   description="Additive resynthesis optimized with Basin Hopping using KL divergence."
   audio="/rendered_audio/optimized_output_additive_parm_bh_kl_20250911-155341.wav"
   plot="/rendered_plots/optimized_output_additive_parm_bh_kl_20250911-155341_spectrum.png|/rendered_plots/optimized_output_additive_parm_bh_kl_20250911-155341_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with BH + manhattan"
   description="Additive resynthesis optimized with Basin Hopping using Manhattan distance."
   audio="/rendered_audio/optimized_output_additive_parm_bh_manhattan_20250911-153357.wav"
   plot="/rendered_plots/optimized_output_additive_parm_bh_manhattan_20250911-153357_spectrum.png|/rendered_plots/optimized_output_additive_parm_bh_manhattan_20250911-153357_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with BH + mfcc"
   description="Additive resynthesis optimized with Basin Hopping using MFCC distance."
   audio="/rendered_audio/optimized_output_additive_parm_bh_mfcc_20250911-142054.wav"
   plot="/rendered_plots/optimized_output_additive_parm_bh_mfcc_20250911-142054_spectrum.png|/rendered_plots/optimized_output_additive_parm_bh_mfcc_20250911-142054_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with BH + pearson"
   description="Additive resynthesis optimized with Basin Hopping using Pearson correlation."
   audio="/rendered_audio/optimized_output_additive_parm_bh_pearson_20250911-133118.wav"
   plot="/rendered_plots/optimized_output_additive_parm_bh_pearson_20250911-133118_spectrum.png|/rendered_plots/optimized_output_additive_parm_bh_pearson_20250911-133118_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with BH + spectral_convergence"
   description="Additive resynthesis optimized with Basin Hopping using spectral convergence."
   audio="/rendered_audio/optimized_output_additive_parm_bh_spectral_convergence_20250911-145142.wav"
   plot="/rendered_plots/optimized_output_additive_parm_bh_spectral_convergence_20250911-145142_spectrum.png|/rendered_plots/optimized_output_additive_parm_bh_spectral_convergence_20250911-145142_time.png"
   captions="Spectrogram|Waveform"
%}
