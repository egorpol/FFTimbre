---
layout: default
title: FFTimbre Showcase
---

<link rel="stylesheet" href="{{ '/assets/style.css' | relative_url }}">

# FFTimbre Showcase

## FFT Frame based on bowed cello sound

Source => `tsv/cello_single.tsv`:

{% include tsv_table.html 
   src="/tsv/cello_single.tsv"
   has_header=true
   max_rows=50
   caption="First 50 rows of cello_single.tsv"
%}

{% include sample.html 
   title="Target spectra"
   description="Replace with your description"
   audio="/rendered_audio/additive_from_cello_single_2.0s_20250906-080046.wav"
   plot="/rendered_plots/additive_from_cello_single_2.0s_20250906-095503_spectrum.png|/rendered_plots/additive_from_cello_single_2.0s_20250906-095503_time.png"
   captions="Spectrogram|Waveform"
%}

Results for batch run with 'maxiter': 500 for all optimizer + metric pairs:

{% include sample.html 
   title="Optimized FM with DE + cosine"
   description="Replace with your description"
   audio="/rendered_audio/optimized_output_fm_cello_single_de_cosine_20250904-003735.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_de_cosine_20250904-003735_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_de_cosine_20250904-003735_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DE + euclidean"
   description="Replace with your description"
   audio="/rendered_audio/optimized_output_fm_cello_single_de_euclidean_20250904-004049.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_de_euclidean_20250904-004049_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_de_euclidean_20250904-004049_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DE + itakura_saito"
   description="Replace with your description"
   audio="/rendered_audio/optimized_output_fm_cello_single_de_itakura_saito_20250904-003136.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_de_itakura_saito_20250904-003136_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_de_itakura_saito_20250904-003136_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DE + kl"
   description="Replace with your description"
   audio="/rendered_audio/optimized_output_fm_cello_single_de_kl_20250904-004429.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_de_kl_20250904-004429_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_de_kl_20250904-004429_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DE + manhattan"
   description="Replace with your description"
   audio="/rendered_audio/optimized_output_fm_cello_single_de_manhattan_20250904-004211.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_de_manhattan_20250904-004211_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_de_manhattan_20250904-004211_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DE + mfcc"
   description="Replace with your description"
   audio="/rendered_audio/optimized_output_fm_cello_single_de_mfcc_20250904-002651.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_de_mfcc_20250904-002651_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_de_mfcc_20250904-002651_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DE + pearson"
   description="Replace with your description"
   audio="/rendered_audio/optimized_output_fm_cello_single_de_pearson_20250904-002050.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_de_pearson_20250904-002050_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_de_pearson_20250904-002050_time.png"
   captions="Spectrogram|Waveform"
%}

{% include sample.html 
   title="Optimized FM with DE + spectral_convergence"
   description="Replace with your description"
   audio="/rendered_audio/optimized_output_fm_cello_single_de_spectral_convergence_20250904-003453.wav"
   plot="/rendered_plots/optimized_output_fm_cello_single_de_spectral_convergence_20250904-003453_spectrum.png|/rendered_plots/optimized_output_fm_cello_single_de_spectral_convergence_20250904-003453_time.png"
   captions="Spectrogram|Waveform"
%}