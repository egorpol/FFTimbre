# optimization.py
from tqdm import tqdm
from scipy.optimize import differential_evolution, dual_annealing, basinhopping
from synthesis import synthesize_fm_chain
from objectives import compute_mfcc, compute_fft, METRIC_FUNCTIONS, METRIC_TYPE
import numpy as np

def define_objective_function(objective_type, target_data, duration, sample_rate):
    """Creates a unified objective function for the optimizer."""
    metric_func = METRIC_FUNCTIONS[objective_type]
    feature_type = METRIC_TYPE[objective_type]

    def objective(params):
        # 1. Synthesize the signal
        generated_signal = synthesize_fm_chain(params, duration, sample_rate)

        # 2. Extract the relevant feature
        if feature_type == 'mfcc':
            generated_feature = compute_mfcc(generated_signal, sample_rate)
        elif feature_type == 'spectrum':
            generated_feature, _ = compute_fft(generated_signal, sample_rate)
        else:
            raise ValueError(f"Feature type for {objective_type} not defined.")

        # --- START OF FIX ---
        # 3. Normalize the generated feature to the same scale as the target before comparison
        # This is CRITICAL for distance-based metrics.
        max_val = np.max(generated_feature)
        if max_val > 0:
            generated_feature /= max_val
        # The target_data (for spectrum) is already a sparse [0,1] array, so they are now comparable.
        # --- END OF FIX ---

        # 4. Calculate and return the distance
        return metric_func(generated_feature, target_data)

    return objective

def run_optimization(method, bounds, objective_func, max_iters):
    """
    Runs the selected optimization algorithm.
    """
    error_history = []
    pbar = tqdm(total=max_iters, unit='iter')

    def callback(xk, convergence=None):
        # Note: Basinhopping's callback works differently. This is a simplification.
        error_history.append(objective_func(xk))
        pbar.update(1)

    if method == 'differential_evolution':
        result = differential_evolution(
            objective_func,
            bounds,
            maxiter=max_iters,
            popsize=15,
            tol=1e-7,
            callback=callback,
            disp=False
        )
    elif method == 'dual_annealing':
        # Dual annealing callback is called at each temperature step, not each iteration
        # The progress bar will be less granular.
        result = dual_annealing(
            objective_func,
            bounds,
            maxiter=max_iters,
            callback=lambda x, f, context: pbar.update(1)
        )
        # We must rebuild error history manually for dual_annealing if needed,
        # as its callback doesn't provide it easily. For simplicity, we omit it here.
    else:
        raise ValueError(f"Optimization method '{method}' not supported.")
    
    pbar.close()
    
    # For differential_evolution, the history is captured by the callback.
    # For others, you might need a wrapper class to store history if the callback is limited.
    return result, error_history