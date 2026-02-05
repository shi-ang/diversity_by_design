import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def systematic_variation(ptb_shifts, avg_ptb_shift):
    """
    Calculate the average cosine similarity between perturbation-specific shifts 
    and the average perturbation effect.
    
    :param ptb_shifts: perturbation shifts matrix of shape (n_perturbations, n_genes)
    :param avg_ptb_shift: average perturbation shift vector of shape (n_genes,)
    """
    similarities = cosine_similarity(ptb_shifts, avg_ptb_shift.reshape(1, -1)).flatten()
    return float(np.mean(similarities))


def intra_data_correlation(
        data
    ):
    """
    Calculate the average pairwise Pearson correlation among samples in the dataset.
    
    :param data: data matrix of shape (n_cells, n_genes)
    """
    # TODO: might want to improve the efficiency using anndata
    corr_matrix = np.corrcoef(data)
    n = corr_matrix.shape[0]
    # Extract upper triangle without diagonal
    upper_tri_indices = np.triu_indices(n, k=1)
    upper_tri_values = corr_matrix[upper_tri_indices]
    return float(np.mean(upper_tri_values))