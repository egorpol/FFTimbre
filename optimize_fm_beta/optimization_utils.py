import numpy as np
from tqdm import tqdm
from scipy.optimize import differential_evolution, dual_annealing, basinhopping

def load_data(file_path):
    import pandas as pd
    df = pd.read_csv(file_path, sep='\t')
    frequencies = df['Frequency (Hz)'].values
    amplitudes = df['Amplitude'].values
    amplitudes /= np.max(amplitudes)
    return frequencies, amplitudes

def generate_target_mfcc(frequencies, amplitudes, duration, sample_rate, sine_wave, compute_mfcc):
    target_signal = np.zeros(int(sample_rate * duration))
    for freq, amp in zip(frequencies, amplitudes):
        target_signal += sine_wave(freq, amp, duration, sample_rate)
    target_signal /= np.max(np.abs(target_signal))
    return compute_mfcc(target_signal, sample_rate)

def define_objective_function(objective_type, target_mfcc_mean, compute_objective):
    return lambda params, target_freqs, target_amps, duration, sample_rate: compute_objective(
        params, target_freqs, target_amps, duration, sample_rate, objective_type, target_mfcc_mean)

def run_optimization(optimization_method, bounds, objective_function, frequencies, amplitudes, duration, sample_rate, n_generations):
    pbar = tqdm(total=n_generations, unit=' iteration', bar_format='{l_bar}{bar} {n}/{total} [ETA: {remaining}, Elapsed: {elapsed}]')
    error_history = []
    
    def callback(param, f=None, accept=None):
        error = objective_function(param, frequencies, amplitudes, duration, sample_rate)
        error_history.append(error)
        pbar.update(1)
        return False
    
    result = None
    if optimization_method == 'differential_evolution':
        result = differential_evolution(
            objective_function, 
            bounds, 
            args=(frequencies, amplitudes, duration, sample_rate), 
            strategy='best1bin', 
            maxiter=n_generations, 
            popsize=10, 
            tol=1e-6, 
            mutation=(0.5, 1), 
            recombination=0.7, 
            callback=callback)
    elif optimization_method == 'dual_annealing':
        result = dual_annealing(
            objective_function, 
            bounds, 
            args=(frequencies, amplitudes, duration, sample_rate),
            maxiter=n_generations,
            initial_temp=5230.0,
            restart_temp_ratio=2e-5,
            visit=5,
            accept=-5.0,
            maxfun=1e7,
            no_local_search=False,
            callback=lambda x, f, context: callback(x, f)
        )
    elif optimization_method == 'basinhopping':
        def bounded_objective(params):
            params_bounded = np.clip(params, [b[0] for b in bounds], [b[1] for b in bounds])
            return objective_function(params_bounded, frequencies, amplitudes, duration, sample_rate)
        
        x0 = np.array([b[0] + (b[1] - b[0]) * np.random.random() for b in bounds])
        
        result = basinhopping(
            bounded_objective,
            x0,
            niter=n_generations,
            T=1.0,
            stepsize=0.5,
            minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds},
            callback=callback
        )
    
    pbar.close()
    return result, error_history

import numpy as np
import matplotlib.pyplot as plt

def plot_time_domain_signal(combined_signal, sample_count=1000):
    plt.figure(figsize=(12, 6))
    plt.plot(combined_signal[:sample_count])  # Plot the first `sample_count` samples
    plt.title('Optimized Combined Signal in Time Domain')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.show()

def plot_frequency_domain_signal(fft_freqs, fft_result_np, frequencies, amplitudes):
    plt.figure(figsize=(12, 6))
    plt.plot(fft_freqs[:len(fft_freqs)//2], fft_result_np[:len(fft_result_np)//2], label='Optimized Spectrum')
    plt.scatter(frequencies, amplitudes, color='red', label='Target Spectrum', zorder=5)
    plt.title('Frequency Spectrum of Optimized Combined Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

def plot_error_history(error_history):
    plt.figure(figsize=(12, 6))
    plt.plot(error_history)
    plt.title('Error Rate Progression')
    plt.xlabel('Generation')
    plt.ylabel('Distance')
    plt.grid(True)
    plt.show()
