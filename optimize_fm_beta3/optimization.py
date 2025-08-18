# optimization.py
from tqdm.auto import tqdm
from scipy.optimize import differential_evolution, dual_annealing, basinhopping
from synthesis import synthesize_fm_chain
from objectives import compute_mfcc, compute_fft, METRIC_FUNCTIONS, METRIC_TYPE
import numpy as np
from types import SimpleNamespace # <-- Import this
import cma # <-- Import this
import warnings

def define_objective_function(
    objective_type,
    target_data,
    duration,
    sample_rate,
    fft_zero_pad: bool = True,
    fft_window: str = 'hann'
):
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
            generated_feature, _ = compute_fft(
                generated_signal,
                sample_rate,
                zero_pad_to_next_pow2=fft_zero_pad,
                window=fft_window
            )
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

        # Enable local minimizer to refine promising solutions
        minimizer_kwargs = {
            'method': 'L-BFGS-B',
            'bounds': bounds,
        }
        params = {
            'maxfun': 1e7,
            'visit': 2.8,
            'initial_temp': 5230,
            'callback': callback_da,
            'minimizer_kwargs': minimizer_kwargs,
            'no_local_search': False,
        }
        if force_full_iters:
            params['maxfun'] = 5e8
        params.update(optimizer_kwargs)

        print(f"Running dual_annealing with params: { {k:v for k,v in params.items() if k != 'callback'} }")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = dual_annealing(objective_func, bounds, maxiter=max_iters, **params)

    elif method == 'cma': # <-- Now correctly placed as an elif
        # For CMA-ES, use scaled variables in [0,1] for better conditioning
        pbar = tqdm(desc=f"Optimizing with {method}", unit='eval', total=max_iters)
        eval_count = 0

        lower = np.array([b[0] for b in bounds], dtype=float)
        upper = np.array([b[1] for b in bounds], dtype=float)
        span = upper - lower

        def to_params(x_scaled: np.ndarray) -> np.ndarray:
            x_scaled = np.clip(x_scaled, 0.0, 1.0)
            return lower + span * x_scaled

        def wrapped_objective_scaled(x_scaled):
            nonlocal eval_count
            eval_count += 1
            if pbar.n < pbar.total:
                pbar.update(1)
            x = to_params(np.asarray(x_scaled))
            return objective_func(x)

        defaults = {
            'sigma0': 0.25,
            'options': {
                'bounds': [[0.0] * len(bounds), [1.0] * len(bounds)],
                'maxfevals': max_iters,
                'CMA_diagonal': True,
                'CMA_active': True,
                'verb_disp': 0,
            },
            'num_restarts': 0,  # manual restarts
        }
        defaults.update(optimizer_kwargs)

        # Initial point in scaled space; allow user to pass x0 (original space) or x0_scaled
        x0 = defaults.pop('x0', None)
        x0_scaled = defaults.pop('x0_scaled', None)
        if x0_scaled is None:
            if x0 is None:
                x0_orig = np.array([np.random.uniform(lo, hi) for lo, hi in bounds], dtype=float)
            else:
                x0_orig = np.asarray(x0, dtype=float)
            # map to [0,1]
            with np.errstate(divide='ignore', invalid='ignore'):
                x0_scaled = (x0_orig - lower) / np.where(span != 0, span, 1.0)
            x0_scaled = np.clip(x0_scaled, 0.0, 1.0)

        # Suggested popsize if not provided
        if 'popsize' not in defaults['options'] and 'popsize' in defaults:
            defaults['options']['popsize'] = defaults.pop('popsize')
        elif 'popsize' not in defaults['options']:
            n = len(bounds)
            defaults['options']['popsize'] = 8 + int(6 * np.log(n))

        sigma0 = defaults.pop('sigma0')
        options = defaults.pop('options')
        num_restarts = int(defaults.pop('num_restarts', 0))

        best_x = None
        best_f = np.inf

        # First run
        print(f"Running cma.fmin2 with options: {options} and sigma0={sigma0}")
        xs, es = cma.fmin2(wrapped_objective_scaled, x0_scaled, sigma0, options=options)
        best_x = es.result.xbest
        best_f = es.result.fbest

        # Manual restarts with increasing population size
        for r in range(num_restarts):
            options['popsize'] = int(options.get('popsize', 0) * 1.5) + 1
            x0_scaled = best_x
            xs, es = cma.fmin2(wrapped_objective_scaled, x0_scaled, sigma0, options=options)
            if es.result.fbest < best_f:
                best_x = es.result.xbest
                best_f = es.result.fbest

        result = SimpleNamespace(x=to_params(best_x), fun=best_f)
        print(f"CMA-ES finished after {eval_count} evaluations. Best f={best_f:.6f}")
    
    else: # <-- The 'else' block is now correctly at the end
        raise ValueError(f"Optimization method '{method}' not supported.")
    
    if pbar:
        pbar.close()
    
    return result, error_history