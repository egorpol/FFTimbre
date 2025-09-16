---
layout: default
title: Voice — "o" Vowel (single frame)
---

<link rel="stylesheet" href="{{ '/assets/style.css' | relative_url }}">

# Voice — "o" Vowel (single frame)

- Source TSV: [`tsv/voice-single2.tsv`]({{ '/tsv/voice-single2.tsv' | relative_url }})
- Back to overview: [FFTimbre Showcase]({{ '/' | relative_url }})

## Legend
- [FFT frame (voice "o" vowel)](#voice-fft-frame)
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

<a id="voice-fft-frame"></a>
## FFT Frame based on voice "o" vowel

{% include tsv_table.html 
   src="/tsv/voice-single2.tsv"
   has_header=true
   max_rows=50
   caption="First 50 rows of voice-single2.tsv"
%}

{% include sample.html 
   title="Target spectra"
   description="Reference spectrogram and waveform for the sung 'o' vowel FFT frame."
   audio="/rendered_audio/additive_from_voice-single2_2.0s_20250908-161359.wav"
   plot="/rendered_plots/additive_from_voice-single2_2.0s_20250908-161359_spectrum.png|/rendered_plots/additive_from_voice-single2_2.0s_20250908-161359_time.png"
   captions="Spectrogram|Waveform"
%}

Results below are from a batch run (`maxiter=500`) across metrics using Differential Evolution.

{% include sample.html 
   title="Optimized FM with DE + cosine"
   description="FM resynthesis optimized with Differential Evolution using cosine similarity."
   audio="/rendered_audio/optimized_output_fm_voice-single2_de_cosine_20250908-163809.wav"
   plot="/rendered_plots/optimized_output_fm_voice-single2_de_cosine_20250908-163809_spectrum.png|/rendered_plots/optimized_output_fm_voice-single2_de_cosine_20250908-163809_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DE + euclidean"
   description="FM resynthesis optimized with Differential Evolution using Euclidean distance."
   audio="/rendered_audio/optimized_output_fm_voice-single2_de_euclidean_20250908-164147.wav"
   plot="/rendered_plots/optimized_output_fm_voice-single2_de_euclidean_20250908-164147_spectrum.png|/rendered_plots/optimized_output_fm_voice-single2_de_euclidean_20250908-164147_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DE + itakura_saito"
   description="FM resynthesis optimized with Differential Evolution using Itakura–Saito divergence."
   audio="/rendered_audio/optimized_output_fm_voice-single2_de_itakura_saito_20250908-163051.wav"
   plot="/rendered_plots/optimized_output_fm_voice-single2_de_itakura_saito_20250908-163051_spectrum.png|/rendered_plots/optimized_output_fm_voice-single2_de_itakura_saito_20250908-163051_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DE + kl"
   description="FM resynthesis optimized with Differential Evolution using KL divergence."
   audio="/rendered_audio/optimized_output_fm_voice-single2_de_kl_20250908-164527.wav"
   plot="/rendered_plots/optimized_output_fm_voice-single2_de_kl_20250908-164527_spectrum.png|/rendered_plots/optimized_output_fm_voice-single2_de_kl_20250908-164527_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DE + manhattan"
   description="FM resynthesis optimized with Differential Evolution using Manhattan distance."
   audio="/rendered_audio/optimized_output_fm_voice-single2_de_manhattan_20250908-164307.wav"
   plot="/rendered_plots/optimized_output_fm_voice-single2_de_manhattan_20250908-164307_spectrum.png|/rendered_plots/optimized_output_fm_voice-single2_de_manhattan_20250908-164307_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DE + mfcc"
   description="FM resynthesis optimized with Differential Evolution using MFCC distance."
   audio="/rendered_audio/optimized_output_fm_voice-single2_de_mfcc_20250908-162141.wav"
   plot="/rendered_plots/optimized_output_fm_voice-single2_de_mfcc_20250908-162141_spectrum.png|/rendered_plots/optimized_output_fm_voice-single2_de_mfcc_20250908-162141_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DE + pearson"
   description="FM resynthesis optimized with Differential Evolution using Pearson correlation."
   audio="/rendered_audio/optimized_output_fm_voice-single2_de_pearson_20250908-161600.wav"
   plot="/rendered_plots/optimized_output_fm_voice-single2_de_pearson_20250908-161600_spectrum.png|/rendered_plots/optimized_output_fm_voice-single2_de_pearson_20250908-161600_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DE + spectral_convergence"
   description="FM resynthesis optimized with Differential Evolution using spectral convergence."
   audio="/rendered_audio/optimized_output_fm_voice-single2_de_spectral_convergence_20250908-163450.wav"
   plot="/rendered_plots/optimized_output_fm_voice-single2_de_spectral_convergence_20250908-163450_spectrum.png|/rendered_plots/optimized_output_fm_voice-single2_de_spectral_convergence_20250908-163450_time.png"
   captions="Spectrogram|Waveform"
%}

Results below are from a batch run (`maxiter=500`) across metrics using Dual Annealing.

{% include sample.html 
   title="Optimized FM with DA + cosine"
   description="FM resynthesis optimized with Dual Annealing using cosine similarity."
   audio="/rendered_audio/optimized_output_fm_voice-single2_da_cosine_20250908-163842.wav"
   plot="/rendered_plots/optimized_output_fm_voice-single2_da_cosine_20250908-163842_spectrum.png|/rendered_plots/optimized_output_fm_voice-single2_da_cosine_20250908-163842_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DA + euclidean"
   description="FM resynthesis optimized with Dual Annealing using Euclidean distance."
   audio="/rendered_audio/optimized_output_fm_voice-single2_da_euclidean_20250908-164212.wav"
   plot="/rendered_plots/optimized_output_fm_voice-single2_da_euclidean_20250908-164212_spectrum.png|/rendered_plots/optimized_output_fm_voice-single2_da_euclidean_20250908-164212_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DA + itakura_saito"
   description="FM resynthesis optimized with Dual Annealing using Itakura–Saito divergence."
   audio="/rendered_audio/optimized_output_fm_voice-single2_da_itakura_saito_20250908-163200.wav"
   plot="/rendered_plots/optimized_output_fm_voice-single2_da_itakura_saito_20250908-163200_spectrum.png|/rendered_plots/optimized_output_fm_voice-single2_da_itakura_saito_20250908-163200_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DA + kl"
   description="FM resynthesis optimized with Dual Annealing using KL divergence."
   audio="/rendered_audio/optimized_output_fm_voice-single2_da_kl_20250908-164557.wav"
   plot="/rendered_plots/optimized_output_fm_voice-single2_da_kl_20250908-164557_spectrum.png|/rendered_plots/optimized_output_fm_voice-single2_da_kl_20250908-164557_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DA + manhattan"
   description="FM resynthesis optimized with Dual Annealing using Manhattan distance."
   audio="/rendered_audio/optimized_output_fm_voice-single2_da_manhattan_20250908-164329.wav"
   plot="/rendered_plots/optimized_output_fm_voice-single2_da_manhattan_20250908-164329_spectrum.png|/rendered_plots/optimized_output_fm_voice-single2_da_manhattan_20250908-164329_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DA + mfcc"
   description="FM resynthesis optimized with Dual Annealing using MFCC distance."
   audio="/rendered_audio/optimized_output_fm_voice-single2_da_mfcc_20250908-162743.wav"
   plot="/rendered_plots/optimized_output_fm_voice-single2_da_mfcc_20250908-162743_spectrum.png|/rendered_plots/optimized_output_fm_voice-single2_da_mfcc_20250908-162743_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DA + pearson"
   description="FM resynthesis optimized with Dual Annealing using Pearson correlation."
   audio="/rendered_audio/optimized_output_fm_voice-single2_da_pearson_20250908-161640.wav"
   plot="/rendered_plots/optimized_output_fm_voice-single2_da_pearson_20250908-161640_spectrum.png|/rendered_plots/optimized_output_fm_voice-single2_da_pearson_20250908-161640_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DA + spectral_convergence"
   description="FM resynthesis optimized with Dual Annealing using spectral convergence."
   audio="/rendered_audio/optimized_output_fm_voice-single2_da_spectral_convergence_20250908-163520.wav"
   plot="/rendered_plots/optimized_output_fm_voice-single2_da_spectral_convergence_20250908-163520_spectrum.png|/rendered_plots/optimized_output_fm_voice-single2_da_spectral_convergence_20250908-163520_time.png"
   captions="Spectrogram|Waveform"
%}

Results below are from a batch run (`maxiter=500`) across metrics using Basin Hopping.

{% include sample.html 
   title="Optimized FM with BH + cosine"
   description="FM resynthesis optimized with Basin Hopping using cosine similarity."
   audio="/rendered_audio/optimized_output_fm_voice-single2_bh_cosine_20250908-163937.wav"
   plot="/rendered_plots/optimized_output_fm_voice-single2_bh_cosine_20250908-163937_spectrum.png|/rendered_plots/optimized_output_fm_voice-single2_bh_cosine_20250908-163937_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with BH + euclidean"
   description="FM resynthesis optimized with Basin Hopping using Euclidean distance."
   audio="/rendered_audio/optimized_output_fm_voice-single2_bh_euclidean_20250908-164234.wav"
   plot="/rendered_plots/optimized_output_fm_voice-single2_bh_euclidean_20250908-164234_spectrum.png|/rendered_plots/optimized_output_fm_voice-single2_bh_euclidean_20250908-164234_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with BH + itakura_saito"
   description="FM resynthesis optimized with Basin Hopping using Itakura–Saito divergence."
   audio="/rendered_audio/optimized_output_fm_voice-single2_bh_itakura_saito_20250908-163235.wav"
   plot="/rendered_plots/optimized_output_fm_voice-single2_bh_itakura_saito_20250908-163235_spectrum.png|/rendered_plots/optimized_output_fm_voice-single2_bh_itakura_saito_20250908-163235_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with BH + kl"
   description="FM resynthesis optimized with Basin Hopping using KL divergence."
   audio="/rendered_audio/optimized_output_fm_voice-single2_bh_kl_20250908-164636.wav"
   plot="/rendered_plots/optimized_output_fm_voice-single2_bh_kl_20250908-164636_spectrum.png|/rendered_plots/optimized_output_fm_voice-single2_bh_kl_20250908-164636_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with BH + manhattan"
   description="FM resynthesis optimized with Basin Hopping using Manhattan distance."
   audio="/rendered_audio/optimized_output_fm_voice-single2_bh_manhattan_20250908-164357.wav"
   plot="/rendered_plots/optimized_output_fm_voice-single2_bh_manhattan_20250908-164357_spectrum.png|/rendered_plots/optimized_output_fm_voice-single2_bh_manhattan_20250908-164357_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with BH + mfcc"
   description="FM resynthesis optimized with Basin Hopping using MFCC distance."
   audio="/rendered_audio/optimized_output_fm_voice-single2_bh_mfcc_20250908-162924.wav"
   plot="/rendered_plots/optimized_output_fm_voice-single2_bh_mfcc_20250908-162924_spectrum.png|/rendered_plots/optimized_output_fm_voice-single2_bh_mfcc_20250908-162924_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with BH + pearson"
   description="FM resynthesis optimized with Basin Hopping using Pearson correlation."
   audio="/rendered_audio/optimized_output_fm_voice-single2_bh_pearson_20250908-161726.wav"
   plot="/rendered_plots/optimized_output_fm_voice-single2_bh_pearson_20250908-161726_spectrum.png|/rendered_plots/optimized_output_fm_voice-single2_bh_pearson_20250908-161726_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with BH + spectral_convergence"
   description="FM resynthesis optimized with Basin Hopping using spectral convergence."
   audio="/rendered_audio/optimized_output_fm_voice-single2_bh_spectral_convergence_20250908-163544.wav"
   plot="/rendered_plots/optimized_output_fm_voice-single2_bh_spectral_convergence_20250908-163544_spectrum.png|/rendered_plots/optimized_output_fm_voice-single2_bh_spectral_convergence_20250908-163544_time.png"
   captions="Spectrogram|Waveform"
%}

Results below are from a batch run (maxiter=500) across metrics using Differential Evolution.

{% include sample.html 
   title="Optimized Additive with DE + cosine"
   description="Additive resynthesis optimized with Differential Evolution using cosine similarity."
   audio="/rendered_audio/optimized_output_additive_voice-single2_de_cosine_20250911-195212.wav"
   plot="/rendered_plots/optimized_output_additive_voice-single2_de_cosine_20250911-195212_spectrum.png|/rendered_plots/optimized_output_additive_voice-single2_de_cosine_20250911-195212_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with DE + euclidean"
   description="Additive resynthesis optimized with Differential Evolution using Euclidean distance."
   audio="/rendered_audio/optimized_output_additive_voice-single2_de_euclidean_20250911-200540.wav"
   plot="/rendered_plots/optimized_output_additive_voice-single2_de_euclidean_20250911-200540_spectrum.png|/rendered_plots/optimized_output_additive_voice-single2_de_euclidean_20250911-200540_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with DE + itakura_saito"
   description="Additive resynthesis optimized with Differential Evolution using Itakura–Saito divergence."
   audio="/rendered_audio/optimized_output_additive_voice-single2_de_itakura_saito_20250911-191900.wav"
   plot="/rendered_plots/optimized_output_additive_voice-single2_de_itakura_saito_20250911-191900_spectrum.png|/rendered_plots/optimized_output_additive_voice-single2_de_itakura_saito_20250911-191900_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with DE + kl"
   description="Additive resynthesis optimized with Differential Evolution using KL divergence."
   audio="/rendered_audio/optimized_output_additive_voice-single2_de_kl_20250911-203042.wav"
   plot="/rendered_plots/optimized_output_additive_voice-single2_de_kl_20250911-203042_spectrum.png|/rendered_plots/optimized_output_additive_voice-single2_de_kl_20250911-203042_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with DE + manhattan"
   description="Additive resynthesis optimized with Differential Evolution using Manhattan distance."
   audio="/rendered_audio/optimized_output_additive_voice-single2_de_manhattan_20250911-202457.wav"
   plot="/rendered_plots/optimized_output_additive_voice-single2_de_manhattan_20250911-202457_spectrum.png|/rendered_plots/optimized_output_additive_voice-single2_de_manhattan_20250911-202457_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with DE + mfcc"
   description="Additive resynthesis optimized with Differential Evolution using MFCC distance."
   audio="/rendered_audio/optimized_output_additive_voice-single2_de_mfcc_20250911-184140.wav"
   plot="/rendered_plots/optimized_output_additive_voice-single2_de_mfcc_20250911-184140_spectrum.png|/rendered_plots/optimized_output_additive_voice-single2_de_mfcc_20250911-184140_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with DE + pearson"
   description="Additive resynthesis optimized with Differential Evolution using Pearson correlation."
   audio="/rendered_audio/optimized_output_additive_voice-single2_de_pearson_20250911-175126.wav"
   plot="/rendered_plots/optimized_output_additive_voice-single2_de_pearson_20250911-175126_spectrum.png|/rendered_plots/optimized_output_additive_voice-single2_de_pearson_20250911-175126_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with DE + spectral_convergence"
   description="Additive resynthesis optimized with Differential Evolution using spectral convergence."
   audio="/rendered_audio/optimized_output_additive_voice-single2_de_spectral_convergence_20250911-193245.wav"
   plot="/rendered_plots/optimized_output_additive_voice-single2_de_spectral_convergence_20250911-193245_spectrum.png|/rendered_plots/optimized_output_additive_voice-single2_de_spectral_convergence_20250911-193245_time.png"
   captions="Spectrogram|Waveform"
%}

Results below are from a batch run (maxiter=500) across metrics using Dual Annealing.

{% include sample.html 
   title="Optimized Additive with DA + cosine"
   description="Additive resynthesis optimized with Dual Annealing using cosine similarity."
   audio="/rendered_audio/optimized_output_additive_voice-single2_da_cosine_20250911-195353.wav"
   plot="/rendered_plots/optimized_output_additive_voice-single2_da_cosine_20250911-195353_spectrum.png|/rendered_plots/optimized_output_additive_voice-single2_da_cosine_20250911-195353_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with DA + euclidean"
   description="Additive resynthesis optimized with Dual Annealing using Euclidean distance."
   audio="/rendered_audio/optimized_output_additive_voice-single2_da_euclidean_20250911-200706.wav"
   plot="/rendered_plots/optimized_output_additive_voice-single2_da_euclidean_20250911-200706_spectrum.png|/rendered_plots/optimized_output_additive_voice-single2_da_euclidean_20250911-200706_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with DA + itakura_saito"
   description="Additive resynthesis optimized with Dual Annealing using Itakura–Saito divergence."
   audio="/rendered_audio/optimized_output_additive_voice-single2_da_itakura_saito_20250911-192243.wav"
   plot="/rendered_plots/optimized_output_additive_voice-single2_da_itakura_saito_20250911-192243_spectrum.png|/rendered_plots/optimized_output_additive_voice-single2_da_itakura_saito_20250911-192243_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with DA + kl"
   description="Additive resynthesis optimized with Dual Annealing using KL divergence."
   audio="/rendered_audio/optimized_output_additive_voice-single2_da_kl_20250911-203514.wav"
   plot="/rendered_plots/optimized_output_additive_voice-single2_da_kl_20250911-203514_spectrum.png|/rendered_plots/optimized_output_additive_voice-single2_da_kl_20250911-203514_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with DA + manhattan"
   description="Additive resynthesis optimized with Dual Annealing using Manhattan distance."
   audio="/rendered_audio/optimized_output_additive_voice-single2_da_manhattan_20250911-202556.wav"
   plot="/rendered_plots/optimized_output_additive_voice-single2_da_manhattan_20250911-202556_spectrum.png|/rendered_plots/optimized_output_additive_voice-single2_da_manhattan_20250911-202556_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with DA + mfcc"
   description="Additive resynthesis optimized with Dual Annealing using MFCC distance."
   audio="/rendered_audio/optimized_output_additive_voice-single2_da_mfcc_20250911-185431.wav"
   plot="/rendered_plots/optimized_output_additive_voice-single2_da_mfcc_20250911-185431_spectrum.png|/rendered_plots/optimized_output_additive_voice-single2_da_mfcc_20250911-185431_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with DA + pearson"
   description="Additive resynthesis optimized with Dual Annealing using Pearson correlation."
   audio="/rendered_audio/optimized_output_additive_voice-single2_da_pearson_20250911-175737.wav"
   plot="/rendered_plots/optimized_output_additive_voice-single2_da_pearson_20250911-175737_spectrum.png|/rendered_plots/optimized_output_additive_voice-single2_da_pearson_20250911-175737_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with DA + spectral_convergence"
   description="Additive resynthesis optimized with Dual Annealing using spectral convergence."
   audio="/rendered_audio/optimized_output_additive_voice-single2_da_spectral_convergence_20250911-193429.wav"
   plot="/rendered_plots/optimized_output_additive_voice-single2_da_spectral_convergence_20250911-193429_spectrum.png|/rendered_plots/optimized_output_additive_voice-single2_da_spectral_convergence_20250911-193429_time.png"
   captions="Spectrogram|Waveform"
%}

Results below are from a batch run (maxiter=500) across metrics using Basin Hopping.

{% include sample.html 
   title="Optimized Additive with BH + cosine"
   description="Additive resynthesis optimized with Basin Hopping using cosine similarity."
   audio="/rendered_audio/optimized_output_additive_voice-single2_bh_cosine_20250911-200227.wav"
   plot="/rendered_plots/optimized_output_additive_voice-single2_bh_cosine_20250911-200227_spectrum.png|/rendered_plots/optimized_output_additive_voice-single2_bh_cosine_20250911-200227_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with BH + euclidean"
   description="Additive resynthesis optimized with Basin Hopping using Euclidean distance."
   audio="/rendered_audio/optimized_output_additive_voice-single2_bh_euclidean_20250911-202234.wav"
   plot="/rendered_plots/optimized_output_additive_voice-single2_bh_euclidean_20250911-202234_spectrum.png|/rendered_plots/optimized_output_additive_voice-single2_bh_euclidean_20250911-202234_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with BH + itakura_saito"
   description="Additive resynthesis optimized with Basin Hopping using Itakura–Saito divergence."
   audio="/rendered_audio/optimized_output_additive_voice-single2_bh_itakura_saito_20250911-192936.wav"
   plot="/rendered_plots/optimized_output_additive_voice-single2_bh_itakura_saito_20250911-192936_spectrum.png|/rendered_plots/optimized_output_additive_voice-single2_bh_itakura_saito_20250911-192936_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with BH + kl"
   description="Additive resynthesis optimized with Basin Hopping using KL divergence."
   audio="/rendered_audio/optimized_output_additive_voice-single2_bh_kl_20250911-204513.wav"
   plot="/rendered_plots/optimized_output_additive_voice-single2_bh_kl_20250911-204513_spectrum.png|/rendered_plots/optimized_output_additive_voice-single2_bh_kl_20250911-204513_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with BH + manhattan"
   description="Additive resynthesis optimized with Basin Hopping using Manhattan distance."
   audio="/rendered_audio/optimized_output_additive_voice-single2_bh_manhattan_20250911-202805.wav"
   plot="/rendered_plots/optimized_output_additive_voice-single2_bh_manhattan_20250911-202805_spectrum.png|/rendered_plots/optimized_output_additive_voice-single2_bh_manhattan_20250911-202805_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with BH + mfcc"
   description="Additive resynthesis optimized with Basin Hopping using MFCC distance."
   audio="/rendered_audio/optimized_output_additive_voice-single2_bh_mfcc_20250911-191634.wav"
   plot="/rendered_plots/optimized_output_additive_voice-single2_bh_mfcc_20250911-191634_spectrum.png|/rendered_plots/optimized_output_additive_voice-single2_bh_mfcc_20250911-191634_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with BH + pearson"
   description="Additive resynthesis optimized with Basin Hopping using Pearson correlation."
   audio="/rendered_audio/optimized_output_additive_voice-single2_bh_pearson_20250911-181143.wav"
   plot="/rendered_plots/optimized_output_additive_voice-single2_bh_pearson_20250911-181143_spectrum.png|/rendered_plots/optimized_output_additive_voice-single2_bh_pearson_20250911-181143_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized Additive with BH + spectral_convergence"
   description="Additive resynthesis optimized with Basin Hopping using spectral convergence."
   audio="/rendered_audio/optimized_output_additive_voice-single2_bh_spectral_convergence_20250911-194946.wav"
   plot="/rendered_plots/optimized_output_additive_voice-single2_bh_spectral_convergence_20250911-194946_spectrum.png|/rendered_plots/optimized_output_additive_voice-single2_bh_spectral_convergence_20250911-194946_time.png"
   captions="Spectrogram|Waveform"
%}
