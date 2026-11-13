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

The `MSVariables` class implements the alternative momentum-space coarse graining
(Meshulam et al., 2018, arXiv:1812.11904; Nicoletti et al., 2020), which projects each variable onto
a decreasing number of leading principal components of the covariance matrix.
"""

import numpy as np
from scipy.spatial import distance
from scipy.stats import spearmanr
from sklearn.metrics import mutual_info_score

from .utils import covariance_evals_and_evectors
from .verbosity import print_if_full, validate_verbosity, warn_if_verbose

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
    # scipy.stats.spearmanr returns a scalar (not a matrix) when given
    # exactly 2 variables; coarse-graining always passes through N=2 on its
    # last step, so that case must be handled explicitly to keep a
    # consistent (N, N) return shape.
    if X.shape[0] == 2:
        rho = spearmanr(X[0], X[1])[0]
        return np.array([[1.0, rho], [rho, 1.0]])
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
    dist_vector = distance.pdist(X, metric="cosine")
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
    dist_vector = distance.pdist(X, metric="hamming") * num_timesteps
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
# Input Validation
# =========================================================================


def _drop_invalid_rows(
    data_array,
    verbose,
    array_name="binary_array",
    index_attribute="CG_cluster_idx",
    constant_description="constant",
):
    """
    Remove rows containing NaN values and constant (zero-variance) rows.

    Both kinds of rows break the coarse graining (NaN propagates through
    sums and correlations; constant rows have undefined correlations and
    cannot be normalized to unit variance), so they are dropped with a
    warning naming their original row indices.

    Parameters
    ----------
    data_array : ndarray of shape (N, t)
        Time series array where rows are variables and columns are time steps.
    verbose : str
        Verbosity level ('silent', 'warnings' or 'full').
    array_name : str, optional
        Name of the input array, used in the warning messages.
    index_attribute : str, optional
        Name of the attribute that stores the surviving original indices,
        used in the warning messages.
    constant_description : str, optional
        How constant rows are described in the warning message.

    Returns
    -------
    data_array : ndarray of shape (N_valid, t)
        Input array with the invalid rows removed.
    original_idx : ndarray of int, shape (N_valid,)
        Index of every surviving row in the array the caller passed in.
    """
    # original_idx tracks, for every row still present in data_array,
    # its index in the array the caller originally passed in. It is
    # refined below as rows get dropped, so stored indices always point
    # back to original variables.
    original_idx = np.arange(data_array.shape[0])

    row_has_nan = np.any(np.isnan(data_array), axis=1)
    if np.any(row_has_nan):
        dropped = original_idx[row_has_nan]
        warn_if_verbose(
            f"Row(s) {dropped.tolist()} of {array_name} contain NaN values and have "
            f"been excluded from coarse graining; their original indices will not "
            f"appear in {index_attribute}. This commonly happens when a continuous row "
            f"with zero variance is z-score binarized (division by a zero standard "
            f"deviation) upstream. Check your preprocessing/binarization if this is "
            f"unexpected.",
            verbose,
        )
        data_array = data_array[~row_has_nan]
        original_idx = original_idx[~row_has_nan]

    # Identify and remove constant rows (zero variance) before assigning
    # base-level indices, so filtered-out variables never appear in the
    # stored indices and can't be mistaken for surviving variables downstream.
    row_is_constant = np.all(data_array == data_array[:, [0]], axis=1)
    if np.any(row_is_constant):
        dropped = original_idx[row_is_constant]
        warn_if_verbose(
            f"Row(s) {dropped.tolist()} of {array_name} are {constant_description} "
            f"across the time window and have zero variance. These variables have been "
            f"excluded from coarse graining; their original indices will not appear in "
            f"{index_attribute}.",
            verbose,
        )
        data_array = data_array[~row_is_constant]
        original_idx = original_idx[~row_is_constant]

    return data_array, original_idx


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

    verbose : str, optional
        Controls console output during coarse graining. Options are
        'silent' (no warnings), 'warnings' (default; emits warnings for
        dropped variables, e.g. odd variable counts or NaN/zero-variance
        rows), or 'full' (same warnings, for use with the higher-level
        `AnalysisParams.verbose="full"` progress/timing output).
        Default is "warnings".

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

    Examples
    --------
    4 short binary variables, coarse-grained by 1 step. Variables 0 and 1 are
    identical (perfectly correlated) and get paired, as do 2 and 3; each
    merged variable is simply the elementwise sum of its pair:

    >>> import numpy as np
    >>> from prg_toolbox import CGVariables
    >>> binary = np.array([
    ...     [1, 0, 1, 0, 1, 0],
    ...     [1, 0, 1, 0, 1, 0],
    ...     [0, 1, 0, 1, 0, 1],
    ...     [0, 0, 0, 1, 0, 1],
    ... ])
    >>> cg = CGVariables(binary, cluster_method='pearson', rg_steps=1)
    >>> cg.CG_cluster_idx[1]  # which original variables got paired together
    [array([0, 1]), array([2, 3])]
    >>> cg.CG_timeseries[1]  # each row is the elementwise sum of its pair
    array([[2., 0., 2., 0., 2., 0.],
           [0., 1., 0., 2., 0., 2.]])
    """

    _CLUSTER_METHODS = {
        "pearson": _compute_pearson,
        "spearman": _compute_spearman,
        "mutual_information": _compute_mutual_information,
        "cosine": _compute_cosine,
        "hamming": _compute_hamming,
        "random": _compute_random,
    }

    def __init__(
        self, binary_array, cluster_method="pearson", rg_steps=6, verbose="warnings"
    ):
        self.cluster_method = cluster_method.lower()
        if self.cluster_method not in self._CLUSTER_METHODS:
            raise ValueError(
                f"Unknown cluster method. Choose from: {list(self._CLUSTER_METHODS.keys())}"
            )

        validate_verbosity(verbose)
        self.verbose = verbose

        self.rg_steps = rg_steps
        self.time_window = binary_array.shape[1]

        # Process and assign variables dynamically
        CG_timeseries, CG_correlation_matrices, CG_cluster_idx = self.get_CG_variables(
            binary_array, self.cluster_method, self.rg_steps
        )
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
        Each variable is used at most once per coarse-graining step. If the
        number of input variables is odd, one variable cannot be paired; it is
        dropped from this and all subsequent scales, and a warning is raised
        naming its original index lineage.
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

        used = np.zeros(N, dtype=bool)

        # 2. Pair most correlated units
        for i in range(halfN):
            idx = np.unravel_index(np.argmax(corr_matrix), corr_matrix.shape)

            # Sum rows to get the next step's block variable
            new_variables[i, :] = old_variables[idx[0]] + old_variables[idx[1]]

            # Track cluster indices
            newclu_idx[i] = np.hstack((oldclu_idx[idx[0]], oldclu_idx[idx[1]]))
            # Mark which variables were used, just to inform ocasional
            # dropped variables if the verbose level is set to "full".
            used[idx[0]] = True
            used[idx[1]] = True

            # Remove paired rows/columns from future pairs
            corr_matrix[idx[0], :] = -np.inf
            corr_matrix[idx[1], :] = -np.inf
            corr_matrix[:, idx[0]] = -np.inf
            corr_matrix[:, idx[1]] = -np.inf

        if N % 2 != 0:
            dropped_local_idx = int(np.flatnonzero(~used)[0])
            dropped_lineage = np.atleast_1d(oldclu_idx[dropped_local_idx]).tolist()
            print_if_full(
                f"Odd number of variables ({N}) at this coarse-graining step; variable "
                f"{dropped_local_idx} (original indices {dropped_lineage}) has no pair "
                f"and has been dropped from this and all subsequent scales.",
                self.verbose,
            )

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

        Notes
        -----
        Rows containing NaN values (typically from z-score binarizing a
        zero-variance continuous row upstream) and constant (zero-variance)
        rows are both dropped before coarse graining begins, each with a
        warning naming the affected original row indices. Dropped indices
        never appear in `clu_idx`.
        """

        # Drop NaN rows before validating binarization, so a preprocessing
        # artifact produces a warning instead of a "not binarized" error.
        binary_array, original_idx = _drop_invalid_rows(
            binary_array,
            self.verbose,
            array_name="binary_array",
            index_attribute="CG_cluster_idx",
            constant_description="constant (all 0s or all 1s)",
        )

        # Validation checks
        if not self.is_binary_matrix(binary_array):
            raise ValueError(
                "Data time series should contain binarized values for real space PRG analysis."
            )
        if binary_array.shape[0] < 2**rg_steps:
            raise ValueError(
                f"Number of variables in data ({binary_array.shape[0]}) is less than required size ({2**self.rg_steps})."
            )
        if binary_array.shape[0] > binary_array.shape[1]:
            warn_if_verbose(
                "Number of variables is greater than number of samples. Correlation measurements may be unreliable.",
                self.verbose,
            )

        total_steps = rg_steps + 1
        CG_var = [None] * total_steps
        corr_matrix = [None] * total_steps
        clu_idx = [None] * total_steps

        # Establish base step (k = 0)
        CG_var[0] = binary_array.astype(np.float64)
        corr_matrix[0] = self._CLUSTER_METHODS[cluster_method](CG_var[0])
        clu_idx[0] = original_idx

        # Execute coarse-graining flow across scales
        for i in range(1, total_steps):
            CG_var[i], corr_matrix[i], clu_idx[i] = self.coarse_grain(
                CG_var[i - 1], clu_idx[i - 1], cluster_method
            )

        return CG_var, corr_matrix, clu_idx


# =========================================================================
# MOMENTUM-SPACE PRG COARSE GRAINING WRAPPER CLASS
# =========================================================================


class MSVariables:
    """
    Wrapper class for performing momentum-space PRG coarse graining on time series data.

    This class applies the momentum-space coarse graining introduced by
    Meshulam et al. (2018, arXiv:1812.11904) and analyzed by Nicoletti et al.
    (2020), who give it in the form used here (their Eqs. 6-7). Instead of merging
    pairs of correlated variables, every variable is projected onto the K
    principal components of the covariance matrix with the largest
    eigenvalues, discarding the fine-grained (small-variance) modes:

        phi_i^(K) = z_i * sum_j sum_{mu <= K} u_{i mu} u_{j mu} (x_j - <x_j>),

    where u_mu are the covariance eigenvectors ordered by decreasing
    eigenvalue and z_i is chosen so that each phi_i^(K) has unit variance.
    At coarse-graining step k the cutoff is K = N / 2^k, so that step 0
    (K = N, all modes kept) gives the z-scored original variables.

    Unlike real-space coarse graining, the number of variables does not
    shrink with k: every step returns N variables. To keep memory usage at
    a single copy of the data, the coarse-grained variables are computed on
    demand by `get_MS_variables` rather than stored for every step.

    Parameters
    ----------
    data_array : numpy array of float
        Time series data of shape (N, t), where N is the number of variables
        and t is the number of time samples. Binary or continuous values are
        accepted.

    rg_steps : int, optional
        Number of coarse graining steps to apply. Step k keeps the
        K = N / 2^k leading principal components, so `rg_steps` must satisfy
        2^rg_steps <= N. Must be a non-negative integer. Default is 6.

    verbose : str, optional
        Controls console output. Options are 'silent' (no warnings),
        'warnings' (default; emits warnings for dropped NaN or constant rows)
        or 'full'. Default is "warnings".

    Attributes
    ----------
    rg_steps : int
        Number of coarse graining steps.
    time_window : int
        Total length of temporal observation slices (t).
    cutoffs : ndarray of int, shape (rg_steps + 1,)
        Number of principal components K = N / 2^k kept at each step k.
    eigenvalues : ndarray of float, shape (N,)
        Covariance eigenvalues in decreasing order.
    eigenvectors : ndarray of float, shape (N, N)
        Covariance eigenvectors (one per column), ordered as `eigenvalues`.
    original_idx : ndarray of int, shape (N,)
        Index, in the array passed by the caller, of each variable kept
        after dropping NaN and constant rows.

    Examples
    --------
    Step 0 keeps all modes and returns the z-scored data; later steps keep
    only the leading modes but still return one variable per original unit:

    >>> import numpy as np
    >>> from prg_toolbox import MSVariables
    >>> rng = np.random.default_rng(0)
    >>> binary = (rng.random((16, 200)) < 0.3).astype(int)
    >>> ms = MSVariables(binary, rg_steps=3)
    >>> ms.cutoffs
    array([16,  8,  4,  2])
    >>> ms.get_MS_variables(2).shape  # 16 variables built from 4 modes
    (16, 200)
    >>> bool(np.allclose(ms.get_MS_variables(2).std(axis=1), 1.0))
    True
    """

    def __init__(self, data_array, rg_steps=6, verbose="warnings"):
        validate_verbosity(verbose)
        self.verbose = verbose

        if not isinstance(rg_steps, (int, np.integer)) or rg_steps < 0:
            raise ValueError(
                f"rg_steps must be a non-negative integer, got {rg_steps!r}."
            )
        self.rg_steps = rg_steps

        data_array = np.asarray(data_array, dtype=np.float64)
        if data_array.ndim != 2:
            raise ValueError(
                f"data_array must be a 2D array of shape (N, t), got {data_array.ndim}D."
            )
        self.time_window = data_array.shape[1]

        data_array, self.original_idx = _drop_invalid_rows(
            data_array,
            self.verbose,
            array_name="data_array",
            index_attribute="original_idx",
        )

        N = data_array.shape[0]
        if N < 2**rg_steps:
            raise ValueError(
                f"Number of variables in data ({N}) is less than required size ({2**rg_steps})."
            )
        if N > data_array.shape[1]:
            warn_if_verbose(
                "Number of variables is greater than number of samples. Covariance eigenvectors may be unreliable.",
                self.verbose,
            )

        self.cutoffs = np.array([N // 2**k for k in range(rg_steps + 1)])
        self.eigenvalues, self.eigenvectors = covariance_evals_and_evectors(data_array)
        self._centered_data = data_array - np.mean(data_array, axis=1, keepdims=True)

    def get_MS_variables(self, k):
        """
        Compute the momentum-space coarse-grained variables at step k.

        Projects the mean-subtracted data onto the K = N / 2^k leading
        covariance eigenvectors and rescales every variable to unit variance.

        Parameters
        ----------
        k : int
            Coarse graining step, between 0 and `rg_steps`.

        Returns
        -------
        ndarray of float, shape (N, t)
            Coarse-grained variables phi^(K). Rows whose projection has zero
            variance (they lie entirely in the discarded modes) cannot be
            normalized and are returned as NaN, with a warning.

        Notes
        -----
        The projection matrix P = U_K U_K^T is never built explicitly:
        U_K (U_K^T x) costs O(N K t) instead of O(N^2 t).
        """
        if not isinstance(k, (int, np.integer)) or not 0 <= k <= self.rg_steps:
            raise ValueError(
                f"Step k must be an integer between 0 and {self.rg_steps}, got {k!r}."
            )

        leading_modes = self.eigenvectors[:, : self.cutoffs[k]]
        projected = leading_modes @ (leading_modes.T @ self._centered_data)

        std = np.std(projected, axis=1, keepdims=True)
        # Tolerance relative to the data scale, since round-off leaves a
        # tiny nonzero std for rows that lie entirely in discarded modes.
        zero_variance = std[:, 0] <= 1e-10 * np.max(std)
        if np.any(zero_variance):
            warn_if_verbose(
                f"At step {k} (K = {self.cutoffs[k]}), variable(s) "
                f"{self.original_idx[zero_variance].tolist()} have zero variance after "
                f"projection onto the leading modes and are returned as NaN.",
                self.verbose,
            )
            std[zero_variance] = np.nan

        return projected / std
