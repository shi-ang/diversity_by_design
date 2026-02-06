import argparse
import numpy as np
import pandas as pd
import os
import time
import tempfile
import warnings
from datetime import datetime
import multiprocessing
from tqdm import tqdm
from scipy import stats, sparse

import anndata as ad
import scanpy as sc
from anndata.experimental import AnnCollection

from .util import est_cost, systematic_variation, intra_corr_accumulator, sum_and_sumsq
from metrics.perturbation_effect.pearson import pearson_pert
from metrics.perturbation_effect.perturbation_discrimination_score import compute_pds
from metrics.perturbation_effect.r_square import r2_score_pert
from metrics.reconstruction.mean_error import mean_error_pert
from dgp.synthetic_one import synthetic_DGP

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


def evaluation(
        x_pred,
        x_obs,
        mu_pred,
        mu_obs,
        mu_control_obs,
        mu_pool_obs,
        DEGs_stats,
        DEGs_by_DGP,
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
    :param DEGs_stats: list of differentially expressed genes masks for each perturbation, from statistical testing
    :param DEGs_by_DGP: list of masks indicating affected genes for each perturbation, from data generation process
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
    if DEGs_by_DGP:
        for DEGs in DEGs_by_DGP:
            any_affected_calculate = np.logical_or(any_affected_calculate, DEGs)
    
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
        DEGs_by_DGP_ptb = DEGs_by_DGP[ptb_idx]
        DEGs_stats_ptb = DEGs_stats[ptb_idx]  # Get the DEGs mask for current perturbation
        
        mu_obs_ptb = mu_obs[ptb_idx].astype(np.float32)
        mu_pred_ptb = mu_pred[ptb_idx].astype(np.float32)
        if model != "Control":
            results_tracker['pearson_all'].append(pearson_pert(mu_obs_ptb, mu_pred_ptb, reference=mu_control_obs))
            results_tracker['pearson_affected'].append(pearson_pert(mu_obs_ptb, mu_pred_ptb, reference=mu_control_obs, DEGs=DEGs_by_DGP_ptb))
            results_tracker['pearson_degs'].append(pearson_pert(mu_obs_ptb, mu_pred_ptb, reference=mu_control_obs, DEGs=DEGs_stats_ptb))

        results_tracker['mae_all'].append(mean_error_pert(mu_obs_ptb, mu_pred_ptb, type="absolute"))
        results_tracker['mse_all'].append(mean_error_pert(mu_obs_ptb, mu_pred_ptb, type="squared"))
        results_tracker['mae_affected'].append(mean_error_pert(mu_obs_ptb, mu_pred_ptb, type="absolute", weights=DEGs_by_DGP_ptb.astype(np.float32))) 
        results_tracker['mse_affected'].append(mean_error_pert(mu_obs_ptb, mu_pred_ptb, type="squared", weights=DEGs_by_DGP_ptb.astype(np.float32)))
        results_tracker['mae_degs'].append(mean_error_pert(mu_obs_ptb, mu_pred_ptb, type="absolute", weights=DEGs_stats_ptb.astype(np.float32)))
        results_tracker['mse_degs'].append(mean_error_pert(mu_obs_ptb, mu_pred_ptb, type="squared", weights=DEGs_stats_ptb.astype(np.float32)))

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


def simulate_one_run_numpy(
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
    max_cells_per_chunk: int=2048, # Count-generation chunk size
    ann_batch_size: int=1024, # AnnCollection batch size for scanpy processing
):
    """
    Simulate one experiment using chunked AnnData/AnnCollection processing.
    This avoids materializing the full (cells x genes) matrix in memory.
    """
    # Setup temporary directory for chunked data
    # The directory and its contents will be deleted after use
    with tempfile.TemporaryDirectory(prefix=f"synthetic_trial_{trial_id_for_rng}_", dir="/tmp") as tmp_dir:
        chunk_paths, true_DEGs_generated = synthetic_DGP(
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
            output_dir_for_chunks=tmp_dir,
            max_cells_per_chunk=max_cells_per_chunk,
        )

        # Memmaps keep large (P x G) accumulators off RAM while still supporting ndarray ops.
        pert_sum = np.memmap(
            os.path.join(tmp_dir, "pert_sum.dat"),
            mode="w+",
            dtype=np.float32,
            shape=(P, G),
        )
        pert_sumsq = np.memmap(
            os.path.join(tmp_dir, "pert_sumsq.dat"),
            mode="w+",
            dtype=np.float32,
            shape=(P, G),
        )
        pert_sum[:] = 0.0
        pert_sumsq[:] = 0.0
        pert_counts = np.zeros(P, dtype=np.int64)

        control_sum = np.zeros(G, dtype=np.float64)
        control_sumsq = np.zeros(G, dtype=np.float64)
        control_count = 0

        pool_sum = np.zeros(G, dtype=np.float64)
        pool_count = 0

        total_nonzero = 0
        total_entries = 0
        library_size_sum = 0.0
        total_cells = 0

        intra_corr_running_sum = np.zeros(G, dtype=np.float64)
        intra_corr_total_cells = 0
        intra_corr_invalid = False

        # Read chunk files lazily and iterate in observation batches.
        backed_chunks = [ad.read_h5ad(path, backed="r") for path in chunk_paths]
        try:
            collection = AnnCollection(
                backed_chunks,
                join_vars="inner",
                label="chunk_id",
                keys=[str(i) for i in range(len(backed_chunks))],
                index_unique="-",
            )

            for batch_view, _ in collection.iterate_axis(batch_size=ann_batch_size, axis=0, shuffle=False):
                batch_counts = batch_view.X
                batch_n_cells = batch_counts.shape[0]
                total_cells += batch_n_cells
                total_entries += batch_n_cells * G

                if sparse.issparse(batch_counts):
                    total_nonzero += int(batch_counts.nnz)
                    batch_library_sizes = np.asarray(batch_counts.sum(axis=1)).ravel().astype(np.float64, copy=False)
                else:
                    batch_counts_dense = np.asarray(batch_counts)
                    total_nonzero += int(np.count_nonzero(batch_counts_dense))
                    batch_library_sizes = batch_counts_dense.sum(axis=1, dtype=np.float64)
                library_size_sum += float(batch_library_sizes.sum())

                # Run scanpy preprocessing per batch to avoid full-matrix normalization/log1p.
                batch_adata = ad.AnnData(X=batch_counts.copy())
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Some cells have zero counts")
                    if normalize:
                        sc.pp.normalize_total(batch_adata, target_sum=1e4, inplace=True)
                sc.pp.log1p(batch_adata)
                batch_log = batch_adata.X

                if not intra_corr_invalid:
                    intra_corr_running_sum, rows_added, has_invalid_row = intra_corr_accumulator(
                        batch_log, intra_corr_running_sum
                    )
                    intra_corr_total_cells += rows_added
                    intra_corr_invalid = has_invalid_row

                perturbation_ids = batch_view.obs["perturbation"].to_numpy(dtype=np.int32, copy=False)
                for perturbation_id in np.unique(perturbation_ids):
                    # Aggregate sufficient statistics per perturbation ID.
                    row_mask = perturbation_ids == perturbation_id
                    batch_subset = batch_log[row_mask]
                    subset_count = int(batch_subset.shape[0])
                    subset_sum, subset_sumsq = sum_and_sumsq(batch_subset)

                    if perturbation_id == -1:
                        control_sum += subset_sum
                        control_sumsq += subset_sumsq
                        control_count += subset_count
                        continue

                    if perturbation_id < 0 or perturbation_id >= P:
                        raise ValueError(f"Invalid perturbation id encountered: {perturbation_id}")

                    pert_sum[perturbation_id, :] += subset_sum.astype(np.float32, copy=False)
                    pert_sumsq[perturbation_id, :] += subset_sumsq.astype(np.float32, copy=False)
                    pert_counts[perturbation_id] += subset_count
                    pool_sum += subset_sum
                    pool_count += subset_count
        finally:
            for backed_adata in backed_chunks:
                if getattr(backed_adata, "file", None) is not None:
                    backed_adata.file.close()

        if control_count == 0:
            raise ValueError("No control cells were generated.")
        if pool_count == 0:
            raise ValueError("No perturbation cells were generated.")
        if np.any(pert_counts == 0):
            missing_ids = np.flatnonzero(pert_counts == 0).tolist()
            raise ValueError(f"Missing cells for perturbations: {missing_ids[:10]}")
        
        # Compute control means and stds
        mu_control_float64 = control_sum / control_count
        var_control_float64 = control_sumsq / control_count - np.square(mu_control_float64)
        var_control = np.clip(var_control_float64, 0.0, None).astype(np.float32, copy=False)
        control_std = np.sqrt(var_control, dtype=np.float32)
        mu_control = mu_control_float64.astype(np.float32, copy=False)
        mu_pool = (pool_sum / pool_count).astype(np.float32, copy=False)

        # Compute observed means and DEGs per perturbation
        mu_obs = np.empty((P, G), dtype=np.float32)
        degs_list = []
        for p_idx in range(P):
            mu_pert_float64 = pert_sum[p_idx, :].astype(np.float64, copy=False) / pert_counts[p_idx]
            var_pert_float64 = (
                pert_sumsq[p_idx, :].astype(np.float64, copy=False) / pert_counts[p_idx]
                - np.square(mu_pert_float64)
            )
            mu_obs[p_idx, :] = mu_pert_float64.astype(np.float32, copy=False)
            std_pert = np.sqrt(np.clip(var_pert_float64, 0.0, None)).astype(np.float32, copy=False)

            _, pvals = stats.ttest_ind_from_stats(
                mean1=mu_obs[p_idx, :],
                std1=std_pert,
                nobs1=int(pert_counts[p_idx]),
                mean2=mu_control,
                std2=control_std,
                nobs2=Nk, # NOTE: Left like this for the over estimation of variance
                equal_var=False,  # Welch's, going conservative due to large size differences
            )

            n_degs = int(true_DEGs_generated[p_idx].sum())
            if n_degs > 0:
                top_deg_indices = np.argpartition(pvals, n_degs - 1)[:n_degs]
            else:
                top_deg_indices = np.array([], dtype=np.int32)
            deg_mask = np.zeros(G, dtype=bool)
            deg_mask[top_deg_indices] = True
            degs_list.append(deg_mask)

        pert_sum.flush()
        pert_sumsq.flush()
        del pert_sum, pert_sumsq

        sparsity = 1.0 - (total_nonzero / total_entries) if total_entries > 0 else np.nan
        avg_library_size = library_size_sum / total_cells if total_cells > 0 else np.nan

        if intra_corr_invalid or intra_corr_total_cells < 2:
            intra_corr = np.nan
        else:
            # Mean pairwise correlation from normalized-row sum identity:
            # mean_{i<j}(u_i dot u_j) = (||sum_i u_i||^2 - n) / (n*(n-1)).
            numerator = float(np.dot(intra_corr_running_sum, intra_corr_running_sum) - intra_corr_total_cells)
            denominator = float(intra_corr_total_cells * (intra_corr_total_cells - 1))
            intra_corr = numerator / denominator if denominator > 0 else np.nan
            intra_corr = float(np.clip(intra_corr, -1.0, 1.0))

        sys_var = systematic_variation(
            ptb_shifts=mu_obs,
            avg_ptb_shift=mu_pool - mu_control,
        )

        data_stats = {
            'sparsity': sparsity,
            'average_library_size': avg_library_size,
            'systematic_variation': sys_var,
            'intra_data_correlation': intra_corr,
        }

        all_results = []
        for model in _MODELS:
            start_time = time.time()
            if model == "Control":
                x_pred = None
                mu_pred = np.tile(mu_control, (P, 1))
            elif model == "Average":
                x_pred = None
                mu_pred = np.tile(mu_pool, (P, 1))
            else:
                raise NotImplementedError(f"Model '{model}' is not implemented.")

            model_results = evaluation(
                x_pred=x_pred,
                x_obs=None,
                mu_obs=mu_obs,
                mu_pred=mu_pred,
                mu_control_obs=mu_control,
                mu_pool_obs=mu_pool,
                DEGs_stats=degs_list,
                DEGs_by_DGP=true_DEGs_generated,
                model=model,
            )
            model_results.update({
                'model': model,
                'execution_time': time.time() - start_time,
                **data_stats
            })
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
            'sparsity', 'average_library_size', 'systematic_variation', 'intra_data_correlation',
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


def run_random_sweep(
    n_trials,
    output_dir,
    control_mu=None,
    all_theta=None,
    pert_mu=None,
    num_workers=None,
    use_multiprocessing=True,
    max_cells_per_chunk=2048,
    ann_batch_size=1024,
):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(output_dir, f"random_sweep_results_{timestamp}.csv")
    error_log_file = os.path.join(output_dir, f"error_log_{timestamp}.txt")

    if use_multiprocessing:
        if num_workers is None:
            num_workers = os.cpu_count()
        print(f"Starting AnnData-backed random parameter sweep with {n_trials} trials using {num_workers} worker processes (spawn context).")
    else:
        print(f"Starting AnnData-backed random parameter sweep with {n_trials} trials using sequential execution.")

    tasks_for_pool = []
    for i in range(n_trials):
        params = sample_parameters(_PARAM_RANGES)
        params['max_cells_per_chunk'] = int(max_cells_per_chunk)
        params['ann_batch_size'] = int(ann_batch_size)
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
            print("\nProcessing trials (AnnData-backed version with worker timing):")
            with tqdm(total=n_trials, desc="Running Trials (AnnData-backed)") as pbar:
                for result_from_worker in pool.imap_unordered(_pool_worker_timed, tasks_for_pool):
                    all_results_data += result_from_worker
                    pbar.update(1)
    else:
        init_worker(control_mu, all_theta, pert_mu)
        print("\nProcessing trials (AnnData-backed version with worker timing):")
        with tqdm(total=n_trials, desc="Running Trials (AnnData-backed)") as pbar:
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
    parser.add_argument("--output_dir", type=str, default="results/synthetic_simulations/random_sweep_results", help="Directory to save sweep results")
    parser.add_argument("--n_trials", type=int, default=2, help="Number of trials to run")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of worker processes for multiprocessing")
    parser.add_argument("--multiprocessing", action="store_true", help="Enable multiprocessing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max_cells_per_chunk", type=int, default=2048, help="Maximum cells per generated h5ad chunk")
    parser.add_argument("--ann_batch_size", type=int, default=1024, help="Batch size when iterating AnnCollection")
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    
    # Load fitted parameters from parameter estimation files
    control_params_df = pd.read_csv("results/synthetic_simulations/parameter_estimation/control_fitted_params.csv", index_col=0)
    perturbed_params_df = pd.read_csv("results/synthetic_simulations/parameter_estimation/perturbed_fitted_params.csv", index_col=0)
    all_params_df = pd.read_csv("results/synthetic_simulations/parameter_estimation/all_fitted_params.csv", index_col=0)
    
    print("Using theta estimates from all cells combined")
    
    # Extract parameters for simulation
    main_control_mu_loaded = control_params_df['mu'].values
    main_pert_mu_loaded = perturbed_params_df['mu'].values
    
    # Use theta (n) from all cells estimation
    main_all_theta_loaded = all_params_df['n'].values
    
    print(f"Using {len(main_control_mu_loaded)} genes for simulation.")
    
    # Call the final version of run_random_sweep
    print("Running the sweep...")
    csv_file = run_random_sweep(
        args.n_trials, 
        args.output_dir, 
        control_mu=main_control_mu_loaded, 
        all_theta=main_all_theta_loaded,
        pert_mu=main_pert_mu_loaded,
        num_workers=args.num_workers, # num_worker should be around 0.6 * RAM / MAX_SPACE_PER_WORK
        use_multiprocessing=args.multiprocessing,
        max_cells_per_chunk=args.max_cells_per_chunk,
        ann_batch_size=args.ann_batch_size,
    ) 
    print("\nDone doing the sweep. Plotting results...")

    # Run uv run python simulations/simulation_plots.py
    os.system(f"uv run python analyses/synthetic_simulations/paper_plots.py --results {csv_file}")
