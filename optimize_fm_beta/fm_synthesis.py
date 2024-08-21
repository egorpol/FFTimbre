import numpy as np
from scipy.optimize import differential_evolution, dual_annealing, basinhopping
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from tqdm import tqdm

class FMSynthesizer:
    def __init__(self, sample_rate=44100, duration=1.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.time_vector = np.linspace(0, duration, int(sample_rate * duration))

    def sine_wave(self, frequency, amplitude, modulator=None):
        if modulator is not None:
            return amplitude * np.sin(2 * np.pi * frequency * self.time_vector + modulator)
        return amplitude * np.sin(2 * np.pi * frequency * self.time_vector)

    def fm_modulate(self, carrier_freq, carrier_amp, modulator_signal):
        return carrier_amp * np.sin(2 * np.pi * carrier_freq * self.time_vector + modulator_signal)

    def generate_combined_signal(self, params):
        freq4, amp4 = params[0], params[1]
        freq3, amp3 = params[2], params[3]
        freq2, amp2 = params[4], params[5]
        freq1, amp1 = params[6], params[7]

        mod4 = self.sine_wave(freq4, amp4)
        mod3 = self.fm_modulate(freq3, amp3, mod4)
        mod2 = self.fm_modulate(freq2, amp2, mod3)
        combined_signal = self.fm_modulate(freq1, amp1, mod2)
        
        max_val = np.max(np.abs(combined_signal))
        if max_val > 0:
            combined_signal /= max_val
        
        return combined_signal

class FMOptimizer:
    def __init__(self, synthesizer, objective_type='kullback_leibler_divergence'):
        self.synthesizer = synthesizer
        self.objective_type = objective_type
        self.error_history = []
        self.target_mfcc_mean = None

    def compute_objective(self, params, target_freqs, target_amps):
        combined_signal = self.synthesizer.generate_combined_signal(params)
        
        fft_result = np.abs(np.fft.fft(combined_signal))
        fft_freqs = np.fft.fftfreq(len(fft_result), 1/self.synthesizer.sample_rate)
        
        target_spectrum = np.zeros_like(fft_result)
        for target_freq, target_amp in zip(target_freqs, target_amps):
            closest_index = np.argmin(np.abs(fft_freqs - target_freq))
            target_spectrum[closest_index] = target_amp
        
        epsilon = 1e-10  # Small value to avoid log(0) and division by zero
        
        if self.objective_type == 'kullback_leibler_divergence':
            return np.sum(target_spectrum * np.log((target_spectrum + epsilon) / (fft_result + epsilon)))
        elif self.objective_type == 'itakura_saito':
            return np.sum((target_spectrum + epsilon) / (fft_result + epsilon) - np.log((target_spectrum + epsilon) / (fft_result + epsilon)) - 1)
        elif self.objective_type == 'spectral_convergence':
            return np.linalg.norm(np.abs(fft_result) - np.abs(target_spectrum)) / np.linalg.norm(np.abs(target_spectrum))
        elif self.objective_type == 'cosine_similarity':
            return cosine(np.abs(fft_result), np.abs(target_spectrum))
        elif self.objective_type == 'euclidean_distance':
            return np.linalg.norm(np.abs(fft_result) - np.abs(target_spectrum))
        elif self.objective_type == 'manhattan_distance':
            return np.sum(np.abs(np.abs(fft_result) - np.abs(target_spectrum)))
        elif self.objective_type == 'pearson_correlation_coefficient':
            pearson_corr, _ = pearsonr(np.abs(fft_result), target_spectrum)
            return 1 - pearson_corr
        elif self.objective_type == 'mfcc_distance':
            generated_mfcc_mean = compute_mfcc(combined_signal, self.synthesizer.sample_rate)
            return np.linalg.norm(generated_mfcc_mean - self.target_mfcc_mean)
        else:
            raise ValueError(f"Unknown objective type: {self.objective_type}")

    def run_optimization(self, target_freqs, target_amps, bounds, optimization_method='differential_evolution', n_generations=2500):
        objective_function = lambda params: self.compute_objective(params, target_freqs, target_amps)

        pbar = tqdm(total=n_generations, unit=' iteration', bar_format='{l_bar}{bar} {n}/{total} [ETA: {remaining}, Elapsed: {elapsed}]')

        def callback(param, convergence=None):
            error = objective_function(param)
            self.error_history.append(error)
            pbar.update(1)

        if optimization_method == 'differential_evolution':
            result = differential_evolution(
                objective_function, 
                bounds, 
                strategy='best1bin', 
                maxiter=n_generations, 
                popsize=10, 
                tol=1e-6, 
                mutation=(0.5, 1), 
                recombination=0.7, 
                callback=callback
            )
        
        elif optimization_method == 'dual_annealing':
            minimizer_kwargs = {"method": "L-BFGS-B", "tol": 1e-6}  # Add tol to minimizer_kwargs
        
            result = dual_annealing(
                objective_function,
                bounds,
                maxiter=n_generations,
                callback=lambda x, f, context: callback(x, f),
                minimizer_kwargs=minimizer_kwargs  # Pass minimizer_kwargs with tol
            )


        elif optimization_method == 'basinhopping':
            # Define a local minimizer for basinhopping with tolerance
            minimizer_kwargs = {
                "method": "L-BFGS-B", 
                "bounds": bounds,
                "tol": 1e-6  # Similar to differential_evolution's tol
            }
        
            # Custom callback for basinhopping (note: the callback structure is different)
            def basinhopping_callback(x, f, accept):
                error = objective_function(x)
                self.error_history.append(error)
                pbar.update(1)
        
            # Running basinhopping
            result = basinhopping(
                objective_function,
                x0=np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds]), 
                minimizer_kwargs=minimizer_kwargs, 
                niter=n_generations, 
                callback=basinhopping_callback
            )
            
        pbar.close()
        return result
