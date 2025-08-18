# optimization.py
from tqdm.auto import tqdm
from scipy.optimize import differential_evolution, dual_annealing, basinhopping
from synthesis import synthesize_fm_chain
from objectives import compute_mfcc, compute_fft, METRIC_FUNCTIONS, METRIC_TYPE
import numpy as np
from types import SimpleNamespace # <-- Import this
import cma # <-- Import this

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

def run_optimization(method, bounds, objective_func, max_iters, force_full_iters=False, **optimizer_kwargs):
    """
    Runs the selected optimization algorithm with customizable parameters.
    ... (docstring) ...
    """
    error_history = []
    pbar = None # Initialize pbar to None
    result = None

    # --- CORRECTED IF/ELIF STRUCTURE ---
    if method == 'differential_evolution':
        pbar = tqdm(desc=f"Optimizing with {method}", unit='iter', total=max_iters)
        def callback_de(xk, convergence=None):
            error_history.append(objective_func(xk))
            pbar.update(1)

        params = {'popsize': 15, 'tol': 1e-7, 'atol': 1e-7, 'callback': callback_de, 'disp': False}
        if force_full_iters:
            params.update({'tol': 0.0, 'atol': 0.0})
        params.update(optimizer_kwargs)
        
        print(f"Running differential_evolution with params: { {k:v for k,v in params.items() if k != 'callback'} }")
        result = differential_evolution(objective_func, bounds, maxiter=max_iters, **params)

    elif method == 'dual_annealing':
        pbar = tqdm(desc=f"Optimizing with {method}", unit='iter', total=max_iters)
        g_niter = 0
        def callback_da(x, f, context):
            nonlocal g_niter
            g_niter += 1
            pbar.set_postfix_str(f"Global steps: {g_niter}, Obj: {f:.4f}")
            pbar.update(1)

        params = {'maxfun': 1e7, 'visit': 2.8, 'initial_temp': 5230, 'callback': callback_da}
        if force_full_iters:
            params['maxfun'] = 5e8
        params.update(optimizer_kwargs)

        print(f"Running dual_annealing with params: { {k:v for k,v in params.items() if k != 'callback'} }")
        result = dual_annealing(objective_func, bounds, maxiter=max_iters, **params)

    elif method == 'cma': # <-- Now correctly placed as an elif
        # For CMA-ES, max_iters corresponds to 'maxfevals' (max function evaluations)
        pbar = tqdm(desc=f"Optimizing with {method}", unit='eval', total=max_iters)
        eval_count = 0
        def wrapped_objective(params):
            nonlocal eval_count
            eval_count += 1
            if pbar.n < pbar.total: pbar.update(1)
            # We don't save error history for CMA-ES for simplicity
            return objective_func(params)

        params = {
            'sigma0': 0.25,
            'options': {'bounds': [[b[0] for b in bounds], [b[1] for b in bounds]]}
        }
        params.update(optimizer_kwargs)

        x0 = [np.random.uniform(low, high) for low, high in bounds]
        params['options']['maxfevals'] = max_iters
        
        print(f"Running cma.fmin2 with params: {params}")
        es = cma.fmin2(
            wrapped_objective,
            x0,
            params.pop('sigma0'),
            options=params.pop('options') # Pass options as a keyword argument for clarity
        )[1]

        result = SimpleNamespace(x=es.result.xbest, fun=es.result.fbest)
        print(f"CMA-ES finished after {eval_count} evaluations.")
    
    else: # <-- The 'else' block is now correctly at the end
        raise ValueError(f"Optimization method '{method}' not supported.")
    
    if pbar:
        pbar.close()
    
    return result, error_history