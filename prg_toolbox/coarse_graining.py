"""
Copyright (c) 2026 Daniel Miranda Castro. Licensed under the MIT License.

Real-Space Phenomenological Renormalization Group (PRG) Coarse-Graining.

This module provides the core algorithm that performs real-space coarse graining on 
multivariate binary time series (e.g., neural spike-train grids), following the 
framework described in Meshulam et al. (2019).

The primary engine is the `CGVariables` wrapper class, which stores coarse grained
variables across the recursive PRG steps, along with correlation matrices calculated
and original raw variable indices.

The coarse graining rule can be implemented with a handful of similarity metrics, 
including standard correlation definitions as well as information-theoretic and discrete 
alignment measurements (see below).
"""

import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import spearmanr
from scipy.spatial import distance
import warnings

# =========================================================================
# Similarity Metrics
# =========================================================================

def _compute_pearson(X):
    """
    Compute the pairwise Pearson correlation matrix.

    Parameters
    ----------
    X : ndarray of shape (N, t)
        Time series array where rows are variables and columns are time steps.

    Returns
    -------
    ndarray of shape (N, N)
        Symmetric linear correlation coefficient matrix.
    """
    return np.corrcoef(X)

def _compute_spearman(X):
    """
    Compute the pairwise Spearman rank-order correlation matrix.

    Parameters
    ----------
    X : ndarray of shape (N, t)
        Time series array where rows are variables and columns are time steps.

    Returns
    -------
    ndarray of shape (N, N)
        Symmetric monotonic rank correlation matrix.
    """
    return spearmanr(X, axis=1)[0]

def _compute_cosine(X):
    """
    Compute the pairwise cosine similarity matrix.

    Measures the cosine of the angle between two multi-dimensional row vectors 
    independent of their magnitude.

    Parameters
    ----------
    X : ndarray of shape (N, t)
        Time series array where rows are variables and columns are time steps.

    Returns
    -------
    ndarray of shape (N, N)
        Pairwise angular similarity matrix scaled bounded between [-1.0, 1.0].
    """
    # Vectorized cosine similarity: 1 - pairwise_cosine_distance
    dist_vector = distance.pdist(X, metric='cosine')
    dist_matrix = distance.squareform(dist_vector)
    return 1.0 - dist_matrix

def _compute_hamming(X):
    """
    Compute pairwise matching bit counts using the Hamming metric.

    Calculates the exact total number of synchronized sample points where 
    two states match identically.

    Parameters
    ----------
    X : ndarray of shape (N, t)
        Time series array where rows are variables and columns are time steps.

    Returns
    -------
    ndarray of shape (N, N)
        Matrix indicating the absolute number of overlapping time coordinates.
    """
    # Vectorized Hamming match count: total_timesteps - hamming_distance
    num_timesteps = X.shape[1]
    dist_vector = distance.pdist(X, metric='hamming') * num_timesteps
    dist_matrix = distance.squareform(dist_vector)
    return num_timesteps - dist_matrix

def _compute_mutual_information(X):
    """
    Compute the pairwise Shannon Mutual Information matrix using discrete outcomes.

    Quantifies the non-linear information shared between discrete variables.

    Parameters
    ----------
    X : ndarray of shape (N, t)
        Discrete or binarized array where rows are variables and columns are time steps.

    Returns
    -------
    ndarray of shape (N, N)
        Symmetric mutual information matrix expressed in nats.
    """
    # MI requires discrete integration, but we can optimize the index access
    num_vars = X.shape[0]
    matrix = np.zeros((num_vars, num_vars))
    # We still need loops here due to scikit-learn's API
    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            mi = mutual_info_score(X[i], X[j])
            matrix[i, j] = matrix[j, i] = mi
    return matrix
    
def _compute_random(X):
    """
    Generate a pseudo-random symmetric affinity surrogate matrix.

    Utilized as a control null hypothesis.

    Parameters
    ----------
    X : ndarray of shape (N, t)
        Time series array where rows are variables and columns are time steps.

    Returns
    -------
    ndarray of shape (N, N)
        Symmetric random matrix with values uniformly sampled from [0.0, 1.0).
    """
    num_vars = X.shape[0]
    matrix = np.random.rand(num_vars, num_vars)
    matrix = (matrix + matrix.T) / 2.0
    return matrix




# =========================================================================
# CORE PRG COARSE GRAINING WRAPPER CLASS
# =========================================================================

class CGVariables:
    """
    Wrapper class for performing real-space PRG coarse graining on binary time series data.

    This class applies the real-space coarse graining (PRG) procedure described in
    Meshulam et al. (2019) to multidimensional binary time series. It computes and
    stores the coarse-grained variables, correlation matrices, and cluster indices
    across multiple renormalization group (RG) steps.

    Parameters
    ----------
    binary_array : numpy array of int
        Binary time series data of shape (N, t), where N is the number of variables
        and t is the number of time samples. Entries must be 0 or 1.

    cluster_method : str, optional
        Metric used to define similarity between variables during coarse graining.
        Available options are:
        'pearson', 'spearman', 'mutual_information', 'cosine', 'hamming', 'random'.
        Default is "Pearson".

    rg_steps : int, optional
        Number of renormalization group (coarse graining) steps to apply.
        The initial (uncoarse-grained) data corresponds to step k = 0.
        Default is 6 or clusters with up to 64 variables.
  
    Attributes
    ----------
    cluster_method : str
        String identifying the metric used for pairing variables.
    rg_steps : int
        Number of PRG iterations.
    time_window : int
        Total length of temporal observation slices (t).
    CG_timeseries : list of ndarray
        Coarse-grained variables over steps. Scale `k` has shape (N / 2^k, t).
    CG_correlation_matrices : list of ndarray
        Calculated similarity matrices evaluated at the start of each coarse graining step.
    CG_cluster_idx : list of list of ndarray
        Trace indices mapping lineage back to individual constituent units from scale 0.
    """
    _CLUSTER_METHODS = {
    "pearson": _compute_pearson,
    "spearman": _compute_spearman,
    "mutual_information": _compute_mutual_information,
    "cosine": _compute_cosine,
    "hamming": _compute_hamming,
    "random": _compute_random   
    }

    def __init__(self, binary_array, cluster_method="pearson", rg_steps=6):
        self.cluster_method = cluster_method.lower()
        if self.cluster_method not in self._CLUSTER_METHODS:
            raise ValueError(f"Unknown cluster method. Choose from: {list(self._CLUSTER_METHODS.keys())}")
            
        self.rg_steps = rg_steps
        self.time_window = binary_array.shape[1]

        # Process and assign variables dynamically
        CG_timeseries, CG_correlation_matrices, CG_cluster_idx = self.get_CG_variables(binary_array, self.cluster_method, self.rg_steps)
        self.CG_timeseries = CG_timeseries
        self.CG_correlation_matrices = CG_correlation_matrices
        self.CG_cluster_idx = CG_cluster_idx

    def is_binary_matrix(self, M):
        return np.all((M == 0) | (M == 1))

    def coarse_grain(self, old_variables, oldclu_idx, cluster_method):
        """
        Perform one step of real-space coarse graining following Meshulam et al.(2019).

        This function groups pairs of variables based on a chosen similarity metric
        and constructs new coarse-grained variables by summing the paired time series.
        At each step, the two most similar variables are merged, and the procedure
        is repeated until the number of variables is halved.

        Parameters
        ----------
        old_variables : ndarray of float
            Time series data to be coarse grained, of shape (N, t), where N is the
            number of variables and t is the number of time samples.

        oldclu_idx : list of list of int
            Indices of the original variables composing each variable at the
            current RG step.

        cluster_method : str, optional
            Similarity metric used for coarse graining.
            Available options are:
            'pearson', 'spearman', 'mutual_information', 'cosine',
            'hamming', 'random'.

        Returns
        -------
        new_variables : ndarray of float
            Coarse-grained time series of shape (N/2, t), obtained by summing
            pairs of maximally correlated variables.

        corr_matrix : ndarray of float
            Similarity (correlation) matrix used during the pairing procedure.

        newclu_idx : list of list of int
            Updated indices of the original variables composing each
            coarse-grained variable.

        Notes
        -----
        Each variable is used exactly once per coarse-graining step.
        """
        # 1. Compute similarity matrix
        compute_matrix_fn = self._CLUSTER_METHODS[cluster_method]
        corr_matrix = compute_matrix_fn(old_variables)
        return_matrix = corr_matrix.copy()

        N, t = corr_matrix.shape[0], old_variables.shape[1]
        halfN = N // 2

        new_variables = np.zeros((halfN, t), dtype=np.float64)
        newclu_idx = [[] for _ in range(halfN)]

        # Mask identity entries to avoid self-pairing
        np.fill_diagonal(corr_matrix, -np.inf)

        # 2. Pair most correlated units (Meshulam et al. 2019)
        for i in range(halfN):
            idx = np.unravel_index(np.argmax(corr_matrix), corr_matrix.shape)
            
            # Sum rows to generate the next scale's block variable
            new_variables[i, :] = old_variables[idx[0]] + old_variables[idx[1]]
            
            # Track cluster tracking lineage indices
            newclu_idx[i] = np.hstack((oldclu_idx[idx[0]], oldclu_idx[idx[1]]))
            
            # Eliminate paired rows/columns from future lookup steps
            corr_matrix[idx[0], :] = -np.inf
            corr_matrix[idx[1], :] = -np.inf
            corr_matrix[:, idx[0]] = -np.inf
            corr_matrix[:, idx[1]] = -np.inf

        return new_variables, return_matrix, newclu_idx

    def get_CG_variables(self, binary_array, cluster_method, rg_steps):
        """
        Apply real-space PRG coarse graining to multidimensional binary time series.

        Recursively calls `coarse_grain` to generate the coarse grained variables
        data arrays spanning from step 0 to `self.rg_steps`.

        Parameters
        ----------
        binary_array : ndarray of int
            Binary time series data of shape (N, t), where N is the number of
            variables and t is the number of time samples.

        cluster_method : str, optional
            Similarity metric used for coarse graining.
            Available options are:
            'pearson', 'spearman', 'mutual_information', 'cosine',
            'hamming', 'random'.

        rg_steps : int, optional
            Number of renormalization group (coarse graining) steps to apply.
            Step k = 0 corresponds to the original (uncoarse-grained) data.

        Returns
        -------
        CG_var : list of ndarray of float
            Coarse-grained time series at each RG step.
            Entry k contains an array of shape (N_k, t), where N_k = N / 2^k.

        corr_matrix : list of ndarray of float
            Similarity (correlation) matrices used at each RG step.

        clu_idx : list of ndarray of int
            Indices of the original variables composing each coarse-grained
            variable at every RG step.
        """

        # Validation checks
        if not self.is_binary_matrix(binary_array):
            raise ValueError("Data time series should contain binarized values for real space PRG analysis.")
        if binary_array.shape[0] < 2**rg_steps:
            raise ValueError(f"Number of variables in data ({binary_array.shape[0]}) is less than required size ({2**self.rg_steps}).")
        if binary_array.shape[0] > binary_array.shape[1]:
            warnings.warn("Number of variables is greater than number of samples. Correlation measurements may be unreliable.")

        total_steps = rg_steps + 1
        CG_var = [None] * total_steps
        corr_matrix = [None] * total_steps
        clu_idx = [None] * total_steps

        # Establish base step (k = 0)
        CG_var[0] = binary_array.astype(np.float64)
        corr_matrix[0] = self._CLUSTER_METHODS[cluster_method](CG_var[0])
        clu_idx[0] = np.arange(binary_array.shape[0])

        # Execute coarse-graining flow across scales
        for i in range(1, total_steps):
            CG_var[i], corr_matrix[i], clu_idx[i] = self.coarse_grain(CG_var[i-1], clu_idx[i-1], cluster_method)

        return CG_var, corr_matrix, clu_idx