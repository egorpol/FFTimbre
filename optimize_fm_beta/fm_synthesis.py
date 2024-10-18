import numpy as np
from scipy.optimize import differential_evolution, dual_annealing, basinhopping
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from tqdm import tqdm
import librosa

def compute_mfcc(signal, sample_rate, n_mfcc=20):
    """
    Compute the mean MFCCs for the given signal.
    
    Parameters:
        signal (np.ndarray): The audio signal.
        sample_rate (int): The sample rate of the audio signal.
        n_mfcc (int): Number of MFCC components to compute.
        
    Returns:
        np.ndarray: Mean MFCCs.
    """
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1)

class FMSynthesizer:
    def __init__(self, sample_rate=44100, duration=1.0):
        """
        Initialize the FMSynthesizer.
        
        Parameters:
            sample_rate (int): The sample rate for the synthesized signal.
            duration (float): Duration of the signal in seconds.
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.time_vector = np.linspace(0, duration, int(sample_rate * duration))

    def sine_wave(self, frequency, amplitude, modulator=None):
        """
        Generate a sine wave with optional modulation.
        
        Parameters:
            frequency (float): Frequency of the sine wave.
            amplitude (float): Amplitude of the sine wave.
            modulator (np.ndarray, optional): Modulating signal.
            
        Returns:
            np.ndarray: The generated sine wave.
        """
        modulated_signal = 2 * np.pi * frequency * self.time_vector
        if modulator is not None:
            modulated_signal += modulator
        return amplitude * np.sin(modulated_signal)

    def fm_modulate(self, carrier_freq, carrier_amp, modulator_signal):
        """
        Apply frequency modulation to the carrier signal.
        
        Parameters:
            carrier_freq (float): Frequency of the carrier wave.
            carrier_amp (float): Amplitude of the carrier wave.
            modulator_signal (np.ndarray): Modulating signal.
            
        Returns:
            np.ndarray: The modulated signal.
        """
        return self.sine_wave(carrier_freq, carrier_amp, modulator_signal)

    def generate_combined_signal(self, params):
        """
        Generate a combined FM signal from the given parameters.
        
        Parameters:
            params (list): Parameters for the four oscillators.
            
        Returns:
            np.ndarray: The combined FM signal.
        """
        mod4 = self.sine_wave(params[0], params[1])
        mod3 = self.fm_modulate(params[2], params[3], mod4)
        mod2 = self.fm_modulate(params[4], params[5], mod3)
        combined_signal = self.fm_modulate(params[6], params[7], mod2)
        
        max_val = np.max(np.abs(combined_signal))
        if max_val > 0:
            combined_signal /= max_val
        
        return combined_signal

class FMOptimizer:
    def __init__(self, synthesizer, objective_type='kullback_leibler_divergence'):
        """
        Initialize the FMOptimizer.
        
        Parameters:
            synthesizer (FMSynthesizer): The synthesizer to use for signal generation.
            objective_type (str): The type of objective function to use for optimization.
        """
        self.synthesizer = synthesizer
        self.objective_type = objective_type
        self.error_history = []
        self.target_mfcc_mean = None

    def compute_objective(self, params, target_freqs, target_amps):
        """
        Compute the objective function based on the selected type.
        
        Parameters:
            params (list): Parameters for the synthesizer.
            target_freqs (np.ndarray): Target frequencies.
            target_amps (np.ndarray): Target amplitudes.
            
        Returns:
            float: The computed objective value.
        """
        combined_signal = self.synthesizer.generate_combined_signal(params)
        fft_result = np.abs(np.fft.fft(combined_signal))
        fft_freqs = np.fft.fftfreq(len(fft_result), 1/self.synthesizer.sample_rate)
        
        target_spectrum = np.zeros_like(fft_result)
        for target_freq, target_amp in zip(target_freqs, target_amps):
            closest_index = np.argmin(np.abs(fft_freqs - target_freq))
            target_spectrum[closest_index] = target_amp
        
        epsilon = 1e-10

        if self.objective_type == 'kullback_leibler_divergence':
            return np.sum(target_spectrum * np.log((target_spectrum + epsilon) / (fft_result + epsilon)))
        elif self.objective_type == 'itakura_saito':
            return np.sum((target_spectrum + epsilon) / (fft_result + epsilon) - np.log((target_spectrum + epsilon) / (fft_result + epsilon)) - 1)
        elif self.objective_type == 'spectral_convergence':
            return np.linalg.norm(fft_result - target_spectrum) / np.linalg.norm(target_spectrum)
        elif self.objective_type == 'cosine_similarity':
            return cosine(fft_result, target_spectrum)
        elif self.objective_type == 'euclidean_distance':
            return np.linalg.norm(fft_result - target_spectrum)
        elif self.objective_type == 'manhattan_distance':
            return np.sum(np.abs(fft_result - target_spectrum))
        elif self.objective_type == 'pearson_correlation_coefficient':
            pearson_corr, _ = pearsonr(fft_result, target_spectrum)
            return 1 - pearson_corr
        elif self.objective_type == 'mfcc_distance':
            generated_mfcc_mean = compute_mfcc(combined_signal, self.synthesizer.sample_rate)
            return np.linalg.norm(generated_mfcc_mean - self.target_mfcc_mean)
        else:
            raise ValueError(f"Unknown objective type: {self.objective_type}")

    def run_optimization(self, target_freqs, target_amps, bounds, optimization_method='differential_evolution', n_generations=2500, tol=1e-6):
        """
        Run the optimization process using the specified method.
        
        Parameters:
            target_freqs (np.ndarray): Target frequencies.
            target_amps (np.ndarray): Target amplitudes.
            bounds (list): Bounds for the parameters.
            optimization_method (str): The optimization method to use.
            n_generations (int): Number of generations or iterations.
            tol (float): Tolerance for convergence.
            
        Returns:
            OptimizeResult: The result of the optimization process.
        """
        # Compute target MFCC mean if using 'mfcc_distance' as the objective
        if self.objective_type == 'mfcc_distance':
            target_signal = self.synthesizer.generate_combined_signal([target_freqs[i] for i in range(len(target_freqs))] + [target_amps[i] for i in range(len(target_amps))])
            self.target_mfcc_mean = compute_mfcc(target_signal, self.synthesizer.sample_rate)

        objective_function = lambda params: self.compute_objective(params, target_freqs, target_amps)
        pbar = tqdm(total=n_generations, unit=' iteration', bar_format='{l_bar}{bar} {n}/{total} [ETA: {remaining}, Elapsed: {elapsed}]')

        def callback(param, convergence=None):
            error = objective_function(param)
            self.error_history.append(error)
            pbar.update(1)
    
        result = None
        if optimization_method == 'differential_evolution':
            result = differential_evolution(
                func=objective_function,
                bounds=bounds,
                strategy='best1bin',
                maxiter=n_generations,
                popsize=10,
                mutation=(0.5, 1),
                recombination=0.7,
                callback=callback,
                tol=tol
            )
        elif optimization_method == 'dual_annealing':
            result = dual_annealing(
                objective_function,
                bounds,
                maxiter=n_generations,
                callback=lambda x, f, context: callback(x),
                minimizer_kwargs={"method": "L-BFGS-B", "tol": tol},
                initial_temp=5230,
                restart_temp_ratio=2e-5,
                visit=2.62,
                accept=-5.0,
                maxfun=1e7
            )
        elif optimization_method == 'basinhopping':
            minimizer_kwargs = {
                "method": "L-BFGS-B",
                "bounds": bounds,
                "options": {"ftol": tol, "gtol": tol}
            }
            result = basinhopping(
                objective_function,
                x0=np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds]),
                minimizer_kwargs=minimizer_kwargs,
                niter=n_generations,
                callback=lambda x, f, accept: callback(x)
            )
    
        pbar.close()
        return result