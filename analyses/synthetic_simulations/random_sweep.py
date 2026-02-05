import argparse
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
import multiprocessing
from tqdm import tqdm
from scipy import stats

from .util import systematic_variation, intra_data_correlation
from metrics.perturbation_effect.pearson import pearson_pert
from metrics.perturbation_effect.perturbation_discrimination_score import compute_pds
from metrics.perturbation_effect.r_square import r2_score_pert
from metrics.reconstruction.mean_error import mean_error_pert

# No scanpy or anndata imports needed if we are truly removing them

# Set OpenBLAS threads early if it was found to be helpful, otherwise optional
# os.environ["OPENBLAS_NUM_THREADS"] = "1"

_GLOBAL = {}
_PARAM_RANGES = {
    'G': {'type': 'int', 'min': 1000, 'max': 8192}, # 1000, 8192, g
    'N0': {'type': 'log_int', 'min': 10, 'max': 8192}, # 10, 8192, n_0
    'Nk': {'type': 'log_int', 'min': 10, 'max': 256}, # 10, 256, n_p
    'P': {'type': 'log_int', 'min': 10, 'max': 2000}, # 10, 2000, k
    'p_effect': {'type': 'float', 'min': 0.001, 'max': 0.1}, # 0.001, 0.1, delta
    'effect_factor': {'type': 'float', 'min': 1.2, 'max': 5.0}, # 1.2, 5.0, epsilon
    'B': {'type': 'float', 'min': 0.0, 'max': 2.0}, # 0.0, 2.0, beta
    'mu_l': {'type': 'log_float', 'min': 0.2, 'max': 5.0} # 0.2, 5.0, mu_l
}

_MODELS = ["Control", "Average"]

def nb_cells(mean, l_c, theta, rng): # theta kept as generic parameter name for this utility function
    """
    Generate individual cell profiles from NB distribution
    Returns an array of shape (len(l_c), G)
    """
    # Ensure mean and theta are numpy arrays for element-wise operations
    mean_arr = np.asarray(mean) # size of genes
    theta_arr = np.asarray(theta) # size of genes
    l_c_arr = np.asarray(l_c) # size of cells

    # Correct mean for library size
    lib_size_corrected_mean = np.outer(l_c_arr, mean_arr)

    # Prevent division by zero or negative p if theta + mean is zero or mean is much larger than theta
    # This can happen if means are very low and theta is also low.
    # Add a small epsilon to the denominator to stabilize.
    # Also ensure p is within (0, 1)
    p_denominator = theta_arr + lib_size_corrected_mean
    p_denominator[p_denominator <= 0] = 1e-9 # Avoid zero or negative denominator
    
    p = theta_arr / p_denominator
    p = np.clip(p, 1e-9, 1 - 1e-9) # Ensure p is in a valid range for negative_binomial

    # Negative binomial expects n (number of successes, our theta) to be > 0.
    # And p (probability of success) to be in [0, 1].
    # If theta contains zeros or negatives, np.random.negative_binomial will fail.
    # Assuming theta values are appropriate (positive).

    predicted_counts = rng.negative_binomial(theta_arr, p)
    return predicted_counts


def synthetic_DGP(
    G=10_000,   # number of genes
    N0=3_000,   # number of control cells
    Nk=150,     # number of perturbed cells per perturbation
    P=50,       # number of perturbations
    p_effect=0.01,  # a threshold for fraction of genes affected per perturbation
    effect_factor=2.0,  # effect factor for affected genes, epsilon in the paper
    B=0.0,      # global perturbation bias factor, beta in the paper
    mu_l=1.0,   # mean of log library size
    all_theta=None, # Theta parameter for all cells , size of total number of genes in the real dataset (>= G)
    control_mu=None, # Control mu parameters, size of total number of genes in the real dataset (>= G)
    pert_mu=None, # Perturbed mu parameters, size of total number of genes in the real dataset (>= G)
    trial_id_for_rng=None, # Optional for seeding RNG per trial,
):
    """
    Generate synthetic data parameters for the simulation.
    Returns control_mu, pert_mu, all_theta
    Each of shape (G,)
    """
    # Setup random number generator for this trial
    if trial_id_for_rng is not None:
        rng = np.random.RandomState(trial_id_for_rng)
    else:
        rng = np.random.RandomState(42)
    
    # --- Parameter Preparation with assertions ---
    # Assert that control_mu, pert_mu, and all_theta are provided
    assert control_mu is not None, "control_mu must be provided. None value is not allowed."
    assert pert_mu is not None, "pert_mu must be provided. None value is not allowed."
    assert all_theta is not None, "all_theta must be provided. None value is not allowed."
    # Assert that inputs are already arrays
    assert isinstance(control_mu, np.ndarray), "control_mu must be a numpy array"
    assert isinstance(pert_mu, np.ndarray), "pert_mu must be a numpy array"
    assert isinstance(all_theta, np.ndarray), "all_theta must be a numpy array"
    # Assert that they have the same length
    assert len(control_mu) == len(all_theta), "control_mu and all_theta must have the same length."
    assert len(control_mu) == len(pert_mu), "control_mu and pert_mu must have the same length."
    # Assert that G is not larger than the provided arrays
    assert len(control_mu) >= G, f"G parameter ({G}) cannot be larger than the length of provided arrays ({len(control_mu)})"
    # --- End of assertions ---
    
    # Sample G elements from control_mu and all_theta
    indices = rng.choice(len(control_mu), size=G, replace=False)
    local_control_mu = control_mu[indices]
    local_all_theta = all_theta[indices]  # Use the all-cells theta
    local_pert_mu = pert_mu[indices]

    # --- Data Generation (counts) ---
    # Pre-allocate x_mat for raw counts
    x_mat = np.empty((N0 + P * Nk, G), dtype=np.int32)

    # 1. Sample control cells with bias (B, dispersion set to all_theta from all cells, fixed dispersion assumption)
    lib_size_control = rng.lognormal(mean=mu_l, sigma=0.1714, size=N0) # 0.1714 from all cells of the Norman19 dataset
    x_mat[:N0, :] = nb_cells(mean=local_control_mu, l_c=lib_size_control, theta=local_all_theta, rng=rng)
    
    all_affected_masks = [] 
    current_row = N0
    
    # Define global perturbation bias, this is the terms in brackets for eq 2 in the paper
    delta_b = local_pert_mu - local_control_mu
    local_pert_mu_biased = np.clip(local_control_mu + B * delta_b, 0.0, np.inf)

    # 2. For each perturbation generate the cells
    for _ in range(P):
        # this is eq 4 in the paper
        affected_mask_loop = rng.random(G) < p_effect
        all_affected_masks.append(affected_mask_loop)
        
        mu_k_loop = local_pert_mu_biased.copy()
        if affected_mask_loop.sum() > 0:
            effect_directions = rng.choice([effect_factor, 1.0/effect_factor], size=affected_mask_loop.sum())   # alpha in Eq 2 and 4
            mu_k_loop[affected_mask_loop] *= effect_directions

        lib_size_pert = rng.lognormal(mean=mu_l, sigma=0.1714, size=Nk) # 0.1714 from all cells of the Norman19 dataset
        x_mat[current_row : current_row + Nk, :] = nb_cells(mean=mu_k_loop, l_c=lib_size_pert, theta=local_all_theta, rng=rng)
        current_row += Nk
    
    return x_mat, all_affected_masks


def evaluation(
        x_pred,
        x_obs,
        mu_pred,
        mu_obs,
        mu_control_obs,
        mu_pool_obs,
        DEGs_list,
        all_affected_masks,
        model: str = "Average",
):
    """
    Perform evaluation of the predicted single-cell profiles.
    
    :param x_pred: predicted single-cell profiles matrix, shape (n_cells, n_genes)
    :param x_obs: observed single-cell profiles matrix, shape (n_cells, n_genes)
    :param mu_pred: predicted mean expression matrix, shape (n_perturbations, n_genes)
    :param mu_obs: observed mean expression matrix, shape (n_perturbations, n_genes)
    :param mu_control_obs: observed mean expression matrix for control, shape (n_perturbations, n_genes)
    :param mu_pool_obs: observed mean expression matrix for pooled data, shape (n_perturbations, n_genes)
    :param DEGs_list: list of differentially expressed genes masks for each perturbation
    :param all_affected_masks: list of masks indicating affected genes for each perturbation
    :param model: The prediction model used, affects certain calculations
    """
    n_genes = mu_obs.shape[1]
    n_perts = mu_obs.shape[0]
    # PDS(Perturbation Discrimination Score) calculation
    # PDS-l1 and PDS-l2 are independent of reference, because reference will be cancelled out
    # TODO: check if the scores are the same 
    pds_l1_score = compute_pds(
        true_effects=mu_obs, # equally, mu_obs - hat_mu0
        pred_effects=mu_pred, # equally, mu_pred - hat_mu0
        metric="l1",
    )

    pds_l2_score = compute_pds(
        true_effects=mu_obs,
        pred_effects=mu_pred,
        metric="l2",
    )

    # PSD-cosine uses reference, so we use mu_control as reference
    # TODO: this should not be the same, check it
    pds_cosine_score = compute_pds(
        true_effects=mu_obs - mu_control_obs,
        pred_effects=mu_pred - mu_control_obs,
        metric="cosine",
    )

    # get genes which were affected by ANY perturbation for fraction calculation
    any_affected_calculate = np.zeros(n_genes, dtype=bool)
    if all_affected_masks:
        for mask_item in all_affected_masks:
            any_affected_calculate = np.logical_or(any_affected_calculate, mask_item)
    
    results_tracker = {
        'pearson_all': [],
        'pearson_affected': [],
        'pearson_degs': [],
        'mae_all': [],
        'mae_affected': [],
        'mae_degs': [],
        'mse_all': [],
        'mse_affected': [],
        'mse_degs': [],
    }
    # Calculate metrics per perturbation
    # "_all" is for all genes
    # "_affected" is for the genes that were truly affected in the simulation (like true DEGs)
    # "_degs" is for the genes identified as DEGs by the t-test (statistical DEGs)
    for ptb_idx in range(n_perts):
        current_affected_mask = all_affected_masks[ptb_idx]
        current_degs_mask = DEGs_list[ptb_idx]  # Get the DEGs mask for current perturbation
        
        mu_obs_ptb = mu_obs[ptb_idx].astype(np.float32)
        mu_pred_ptb = mu_pred[ptb_idx].astype(np.float32)
        if model != "Control":
            results_tracker['pearson_all'].append(pearson_pert(mu_obs_ptb, mu_pred_ptb, reference=mu_control_obs))
            results_tracker['pearson_affected'].append(pearson_pert(mu_obs_ptb, mu_pred_ptb, reference=mu_control_obs, DEGs=current_affected_mask))
            results_tracker['pearson_degs'].append(pearson_pert(mu_obs_ptb, mu_pred_ptb, reference=mu_control_obs, DEGs=current_degs_mask))

        results_tracker['mae_all'].append(mean_error_pert(mu_obs_ptb, mu_pred_ptb, type="absolute"))
        results_tracker['mse_all'].append(mean_error_pert(mu_obs_ptb, mu_pred_ptb, type="squared"))
        results_tracker['mae_affected'].append(mean_error_pert(mu_obs_ptb, mu_pred_ptb, type="absolute", weights=current_affected_mask.astype(np.float32))) 
        results_tracker['mse_affected'].append(mean_error_pert(mu_obs_ptb, mu_pred_ptb, type="squared", weights=current_affected_mask.astype(np.float32)))
        results_tracker['mae_degs'].append(mean_error_pert(mu_obs_ptb, mu_pred_ptb, type="absolute", weights=current_degs_mask.astype(np.float32)))
        results_tracker['mse_degs'].append(mean_error_pert(mu_obs_ptb, mu_pred_ptb, type="squared", weights=current_degs_mask.astype(np.float32)))

    results_final = {
        'pearson_all_median': np.nanmedian(results_tracker['pearson_all']) if results_tracker['pearson_all'] else np.nan,
        'pearson_affected_median': np.nanmedian(results_tracker['pearson_affected']) if results_tracker['pearson_affected'] else np.nan,
        'pearson_degs_median': np.nanmedian(results_tracker['pearson_degs']) if results_tracker['pearson_degs'] else np.nan,
        'mae_all_median': np.nanmedian(results_tracker['mae_all']) if results_tracker['mae_all'] else np.nan,
        'mae_affected_median': np.nanmedian(results_tracker['mae_affected']) if results_tracker['mae_affected'] else np.nan,
        'mae_degs_median': np.nanmedian(results_tracker['mae_degs']) if results_tracker['mae_degs'] else np.nan,
        'mse_all_median': np.nanmedian(results_tracker['mse_all']) if results_tracker['mse_all'] else np.nan,
        'mse_affected_median': np.nanmedian(results_tracker['mse_affected']) if results_tracker['mse_affected'] else np.nan,
        'mse_degs_median': np.nanmedian(results_tracker['mse_degs']) if results_tracker['mse_degs'] else np.nan,
        'pds_l1': pds_l1_score,
        'pds_l2': pds_l2_score,
        'pds_cosine': pds_cosine_score,
        'fraction_genes_affected': any_affected_calculate.mean() if n_genes > 0 and hasattr(any_affected_calculate, 'size') and any_affected_calculate.size == n_genes and any_affected_calculate.dtype == bool else np.nan,
    }
    return results_final


def simulate_one_run_numpy( # Renamed to signify it's the numpy-only version
    G=10_000,   # number of genes
    N0=3_000,   # number of control cells
    Nk=150,     # number of perturbed cells per perturbation
    P=50,       # number of perturbations
    p_effect=0.01,  # a threshold for fraction of genes affected per perturbation
    effect_factor=2.0,  # effect factor for affected genes, epsilon in the paper
    B=0.0,      # global perturbation bias factor, beta in the paper
    mu_l=1.0,   # mean of log library size
    all_theta=None, # Theta parameter for all cells , size of total number of genes in the real dataset (>= G)
    control_mu=None, # Control mu parameters, size of total number of genes in the real dataset (>= G)
    pert_mu=None, # Perturbed mu parameters, size of total number of genes in the real dataset (>= G)
    trial_id_for_rng=None, # Optional for seeding RNG per trial,
    model="Average", # The prediction model to use
    normalize=True, # Whether to normalize the data
):
    """
    Simulate an experiment using only NumPy/Pandas for calculations.
    """
    x_mat, all_affected_masks = synthetic_DGP(
        G=G,
        N0=N0,
        Nk=Nk,
        P=P,
        p_effect=p_effect,
        effect_factor=effect_factor,
        B=B,
        mu_l=mu_l,
        all_theta=all_theta,
        control_mu=control_mu,
        pert_mu=pert_mu,
        trial_id_for_rng=trial_id_for_rng,
    )

    # --- Calculate some statistics for the synthetic data ---
    # overall sparsity
    sparsity = np.mean(x_mat <= 1e-8)

    # --- Manual Normalization (Counts per 10k) ---
    library_sizes = x_mat.sum(axis=1)
    # Avoid division by zero for cells with no counts
    library_sizes[library_sizes == 0] = 1 
    avg_library_size = np.mean(library_sizes)
    
    if normalize:
        # Ensure library_sizes is a column vector for broadcasting
        norm_factor = 1e4 / library_sizes[:, np.newaxis]
        x_mat = x_mat.astype(np.float32)
        x_mat = x_mat * norm_factor
        del norm_factor
    del library_sizes

    # --- Log1p Transformation ---
    log_x_mat = np.log1p(x_mat)
    del x_mat

    # --- Metric Calculation on log_norm_mat ---
    # Compute both observed mean and variance of control cells, control block is [0:N0]
    mu_control = log_x_mat[:N0, :].mean(axis=0)
    var_control = log_x_mat[:N0, :].var(axis=0)

    # Compute mean and var for the pooled cells (all non-control cells), pooled block is [N0:]
    mu_pool = log_x_mat[N0:, :].mean(axis=0)

    mu_obs = np.empty((P, G), dtype=np.float32)
    degs_list = []

    # for each perturbation, compute mean and variance of expression
    # and compute p-values for all genes, and determine DEGs based on that
    for p_idx in range(P):
        start = N0 + p_idx * Nk
        end = start + Nk
        
        # Compute mean, var and delta for the current perturbation
        mu_pertk = log_x_mat[start:end, :].mean(axis=0)
        var_pertk = log_x_mat[start:end, :].var(axis=0)
        # delta_obs[p_idx, :] = hat_muk - hat_mu0
        mu_obs[p_idx, :] = mu_pertk

        # a vector of p-values for all genes, for this perturbation
        _, pvals = stats.ttest_ind_from_stats(
                    mean1=mu_pertk,
                    std1=np.sqrt(var_pertk),
                    nobs1=Nk,
                    mean2=mu_control,
                    std2=np.sqrt(var_control),
                    nobs2=Nk, # NOTE: Left like this for the over estimation of variance
                    equal_var=False,  # Welch's, going conservative due to large size differences
                )

        # the number of DEGs is determined by the number of affected genes in the simulation (true number of DEGs)
        # the items of DEGs are the genes with the lowest p-values (not necessarily the true DEGs)
        # Add to DEG list a binary mask of the top all_affected_masks[p_idx].sum() DEGs
        n_degs = all_affected_masks[p_idx].sum()
        # Get indices of genes with lowest p-values
        if n_degs > 0:
            top_deg_indices = np.argpartition(pvals, n_degs - 1)[:n_degs]
        else:
            top_deg_indices = np.array([], dtype=np.int32)
        # Create binary mask for DEGs
        deg_mask = np.zeros(G, dtype=bool)
        deg_mask[top_deg_indices] = True
        degs_list.append(deg_mask)
        
    del mu_pertk, var_pertk

    sys_var = systematic_variation(
        ptb_shifts=mu_obs,
        avg_ptb_shift=mu_pool - mu_control,
    )
    intra_corr = intra_data_correlation(data=log_x_mat)

    data_stats = {
        'sparsity': sparsity,
        'average_library_size': avg_library_size,
        'systematic_variation': sys_var,
        'intra_data_correlation': intra_corr,
    }
    del sparsity, avg_library_size, sys_var, intra_corr


    all_results = []
    for model in _MODELS:
        start_time = time.time()
        # --- Fit the prediction model ---
        if model == "Control":
            x_pred = None
            mu_pred = np.tile(mu_control, (P, 1))
        elif model == "Average":
            x_pred = None
            mu_pred = np.tile(mu_pool, (P, 1))
        else:
            raise NotImplementedError(f"Model '{model}' is not implemented.")

        # evaluation
        model_results = evaluation(
            x_pred=x_pred,
            x_obs=log_x_mat,
            mu_obs=mu_obs,
            mu_pred=mu_pred,
            mu_control_obs=mu_control,
            mu_pool_obs=mu_pool,
            DEGs_list=degs_list,
            all_affected_masks=all_affected_masks,
            model=model,
        )
        model_results.update({
            'model': model,
            'execution_time': time.time() - start_time,
            **data_stats
        })

        # add to result dataframe
        all_results.append(model_results)
    
    return all_results


def sample_parameters(param_ranges): # Unchanged from original
    params = {}
    for param, range_info in param_ranges.items():
        if range_info['type'] == 'int':
            params[param] = np.random.randint(range_info['min'], range_info['max'] + 1)
        elif range_info['type'] == 'float':
            params[param] = np.random.uniform(range_info['min'], range_info['max'])
        elif range_info['type'] == 'log_float':
            log_min = np.log10(range_info['min'])
            log_max = np.log10(range_info['max'])
            params[param] = 10 ** np.random.uniform(log_min, log_max)
        elif range_info['type'] == 'log_int':
            log_min = np.log10(range_info['min'])
            log_max = np.log10(range_info['max'])
            log_value = np.random.uniform(log_min, log_max)
            params[param] = int(round(10 ** log_value))
        elif range_info['type'] == 'fixed':
            params[param] = range_info['value']
    return params

def init_worker(control_mu, all_theta, pert_mu):
    _GLOBAL["control_mu"] = control_mu
    _GLOBAL["all_theta"] = all_theta
    _GLOBAL["pert_mu"] = pert_mu

# Revised _pool_worker to include timing (matches spirit of original)
def _pool_worker_timed(task_info_dict):
    trial_id = task_info_dict['trial_id']
    params_dict = task_info_dict['params_dict']
    control_mu_from_main = _GLOBAL["control_mu"]
    all_theta_from_main  = _GLOBAL["all_theta"]
    pert_mu_from_main    = _GLOBAL["pert_mu"]

    # Add trial_id for RNG seeding within simulate_one_run_numpy
    params_for_sim = params_dict.copy() # Avoid modifying original params_dict
    params_for_sim['trial_id_for_rng'] = trial_id
    params_for_sim['control_mu'] = control_mu_from_main
    params_for_sim['all_theta'] = all_theta_from_main
    params_for_sim['pert_mu'] = pert_mu_from_main
    
    try:
        # Ensure all required keys by simulate_one_run_numpy are in params_for_sim
        # G, N0, Nk, P, p_effect, effect_factor are expected from sample_parameters
        results_per_sim = simulate_one_run_numpy(**params_for_sim)
        
        # Prepare results: original sampled params + metrics + supporting info
        # `params_dict` is the original sampled params.
        final_results_per_sim = []
        for results_per_sim_model in results_per_sim:
            final_results_per_sim.append({
                **params_dict, 
                **results_per_sim_model, 
                'trial_id': trial_id, 
                'status': 'success'}
            )
        return final_results_per_sim
        
    except Exception as e:
        # Define metrics_error_keys locally for safety
        metrics_error_keys_local = { 
            'pearson_all_median', 'pearson_affected_median', 'pearson_degs_median',
            'mae_all_median', 'mae_affected_median', 'mae_degs_median',
            'mse_all_median', 'mse_affected_median', 'mse_degs_median',
            'pds_l1', 'pds_l2', 'pds_cosine',
            'fraction_genes_affected', 
            'model', 'execution_time',
            'sparsity', 'average_library_size', 'systematic_variation', 'intra_sample_correlation',
        }
        metrics_error = {key: np.nan for key in metrics_error_keys_local}

        final_result_error = {
            **params_dict, # original sampled params
            **metrics_error,
            'trial_id': trial_id,
            'status': 'failed',
            'error': str(e)
        }
        return [final_result_error] * len(_MODELS)


def est_cost(params):
    """
    Estimate cost for the computation, based on the number of cells and genes.
    
    :param params: Dictionary containing parameters including G, N0, Nk, and P
    """
    G, N0, Nk, P = params["G"], params["N0"], params["Nk"], params["P"]
    rows = N0 + P * Nk
    return rows * G


# And run_random_sweep needs to use _pool_worker_timed and prepare tasks for it:
def run_random_sweep_final(
    n_trials,
    output_dir,
    control_mu=None,
    all_theta=None,
    pert_mu=None,
    num_workers=None,
    use_multiprocessing=True,
): # Renamed from run_random_sweep
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(output_dir, f"random_sweep_results_{timestamp}.csv")
    error_log_file = os.path.join(output_dir, f"error_log_{timestamp}.txt")

    if use_multiprocessing:
        if num_workers is None:
            num_workers = os.cpu_count()
        print(f"Starting NumPy-based random parameter sweep with {n_trials} trials using {num_workers} worker processes (spawn context).")
    else:
        print(f"Starting NumPy-based random parameter sweep with {n_trials} trials using sequential execution.")

    tasks_for_pool = []
    for i in range(n_trials):
        params = sample_parameters(_PARAM_RANGES)
        tasks_for_pool.append({'trial_id': i, 'params_dict': params})
    # sort tasks by estimated cost in descending order to optimize workload distribution
    tasks_for_pool.sort(key=lambda t: est_cost(t["params_dict"]), reverse=True)

    all_results_data = []
    
    if use_multiprocessing:
        print("Multiprocessing can be memory intensive, so if running into swap, reduce the number of workers.")
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(
            processes=num_workers, 
            initializer=init_worker, 
            initargs=(control_mu, all_theta, pert_mu),
            maxtasksperchild=100,
        ) as pool:
            print("\nProcessing trials (NumPy version with worker timing):")
            with tqdm(total=n_trials, desc="Running Trials (NumPy)") as pbar:
                for result_from_worker in pool.imap_unordered(_pool_worker_timed, tasks_for_pool):
                    all_results_data += result_from_worker
                    pbar.update(1)
    else:
        init_worker(control_mu, all_theta, pert_mu)
        print("\nProcessing trials (NumPy version with worker timing):")
        with tqdm(total=n_trials, desc="Running Trials (NumPy)") as pbar:
            for task in tasks_for_pool:
                result_from_worker = _pool_worker_timed(task)
                all_results_data += result_from_worker
                pbar.update(1)

    results_df = pd.DataFrame(all_results_data)
    
    success_count = results_df[(results_df['status'] == 'success') & (results_df["model"] == "Average")].shape[0] if 'status' in results_df else 0
    failure_count = n_trials - success_count

    if failure_count > 0 and 'status' in results_df: # Ensure 'status' column exists
        print("")
        failed_trials = results_df[results_df['status'] == 'failed']
        # Define metrics_error keys for excluding them from params logging
        metrics_error_keys = { 
            'pearson_all_median', 'pearson_affected_median',
            'mae_all_median', 'mae_affected_median',
            'mse_all_median', 'mse_affected_median',
            'pds_l1', 'pds_l2', 'pds_cosine',
            'fraction_genes_affected', 'sparsity'
        }
        with open(error_log_file, 'a') as f:
            for _, row in failed_trials.iterrows():
                # Ensure 'trial_id' and 'error' exist in row, provide defaults if not
                trial_id_val = int(row.get('trial_id', -1))
                error_val = row.get('error', 'Unknown error')
                
                error_params = {k: v for k, v in row.items() if k not in metrics_error_keys and k not in ['status', 'error', 'trial_id', 'execution_time']}
                f.write(f"Trial {trial_id_val + 1} failed\n")
                f.write(f"Parameters: {str(error_params)}\n")
                f.write(f"Error: {error_val}\n")
                f.write("-" * 80 + "\n")

    if not results_df.empty:
        results_df.to_csv(csv_file, index=False)
        print(f"\nSweep complete. Results saved to '{csv_file}'")
    else:
        print("\nSweep complete. No results to save.")

    print(f"Success: {success_count}/{n_trials} trials")
    print(f"Failed: {failure_count}/{n_trials} trials")
    if failure_count > 0:
        print(f"See error log for details: {error_log_file}")
    return csv_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run random sweep simulations.")
    parser.add_argument("--output_dir", type=str, default="analyses/synthetic_simulations/random_sweep_results", help="Directory to save sweep results")
    parser.add_argument("--n_trials", type=int, default=2, help="Number of trials to run")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of worker processes for multiprocessing")
    parser.add_argument("--multiprocessing", action="store_true", help="Enable multiprocessing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    
    # Load fitted parameters from parameter estimation files
    control_params_df = pd.read_csv("analyses/synthetic_simulations/parameter_estimation/control_fitted_params.csv", index_col=0)
    perturbed_params_df = pd.read_csv("analyses/synthetic_simulations/parameter_estimation/perturbed_fitted_params.csv", index_col=0)
    all_params_df = pd.read_csv("analyses/synthetic_simulations/parameter_estimation/all_fitted_params.csv", index_col=0)
    
    print("Using theta estimates from all cells combined")
    
    # Extract parameters for simulation
    main_control_mu_loaded = control_params_df['mu'].values
    main_pert_mu_loaded = perturbed_params_df['mu'].values
    
    # Use theta (n) from all cells estimation
    main_all_theta_loaded = all_params_df['n'].values
    
    print(f"Using {len(main_control_mu_loaded)} genes for simulation.")
    
    # Call the final version of run_random_sweep
    print("Running the sweep...")
    csv_file = run_random_sweep_final(
        args.n_trials, 
        args.output_dir, 
        control_mu=main_control_mu_loaded, 
        all_theta=main_all_theta_loaded,
        pert_mu=main_pert_mu_loaded,
        num_workers=args.num_workers, # num_worker should be around 0.6 * RAM / MAX_SPACE_PER_WORK
        use_multiprocessing=args.multiprocessing
    ) 
    print("\nDone doing the sweep. Plotting results...")

    # Run uv run python simulations/simulation_plots.py
    os.system(f"uv run python analyses/synthetic_simulations/paper_plots.py --results {csv_file}")
