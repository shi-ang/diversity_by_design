from scipy import sparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def est_cost(params):
    """
    Estimate cost for the computation, based on the number of cells and genes.
    
    :param params: Dictionary containing parameters including G, N0, Nk, and P
    """
    G, N0, Nk, P = params["G"], params["N0"], params["Nk"], params["P"]
    rows = N0 + P * Nk
    return rows * G


def systematic_variation(ptb_shifts, avg_ptb_shift):
    """
    Calculate the average cosine similarity between perturbation-specific shifts 
    and the average perturbation effect.
    
    :param ptb_shifts: perturbation shifts matrix of shape (n_perturbations, n_genes)
    :param avg_ptb_shift: average perturbation shift vector of shape (n_genes,)
    """
    similarities = cosine_similarity(ptb_shifts, avg_ptb_shift.reshape(1, -1)).flatten()
    return float(np.mean(similarities))


def intra_corr_accumulator(matrix, running_sum):
    """
    Update the streaming accumulator for mean pairwise Pearson correlation across cells.
    """
    if sparse.issparse(matrix):
        dense_matrix = matrix.toarray().astype(np.float32, copy=False)
    else:
        dense_matrix = np.asarray(matrix, dtype=np.float32)

    row_means = dense_matrix.mean(axis=1, keepdims=True)
    centered = dense_matrix - row_means
    row_norms = np.linalg.norm(centered, axis=1)
    if np.any(row_norms <= 1e-12):
        return running_sum, dense_matrix.shape[0], True

    normalized_rows = centered / row_norms[:, np.newaxis]
    running_sum += normalized_rows.sum(axis=0, dtype=np.float64)
    return running_sum, dense_matrix.shape[0], False


def sum_and_sumsq(matrix):
    """Return per-gene sums and squared sums for dense/sparse matrices."""
    if sparse.issparse(matrix):
        gene_sum = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64, copy=False)
        gene_sumsq = np.asarray(matrix.multiply(matrix).sum(axis=0)).ravel().astype(np.float64, copy=False)
        return gene_sum, gene_sumsq

    dense_matrix = np.asarray(matrix, dtype=np.float32)
    gene_sum = dense_matrix.sum(axis=0, dtype=np.float64)
    gene_sumsq = np.square(dense_matrix, dtype=np.float64).sum(axis=0, dtype=np.float64)
    return gene_sum, gene_sumsq
