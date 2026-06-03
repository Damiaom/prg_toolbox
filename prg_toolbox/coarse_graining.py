import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import spearmanr
from scipy.spatial import distance
import warnings


class CGVariables:
    def __init__(self, binary_array, cluster_method="pearson", rg_steps=6):
        """
        Wrapper class for performing real-space PRG coarse graining on binary time series data.

        This class applies the real-space coarse graining (PRG) procedure described in
        Meshulam et al. (2019) DOI: https://doi.org/10.1103/PhysRevLett.123.178103 to multidimensional binary time series. It computes and
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
        """
        self.time_window = int(len(binary_array[0,:]))
        self.rg_steps = rg_steps
        self.cluster_method = cluster_method
        CG_timeseries, CG_correlation_matrices, CG_cluster_idx = self.get_CG_variables(binary_array, cluster_method=cluster_method, rg_steps=rg_steps)

        self.CG_timeseries = CG_timeseries
        self.CG_correlation_matrices = CG_correlation_matrices
        self.CG_cluster_idx = CG_cluster_idx


    def is_binary_matrix(self, M):
        """
        Checks if input array is binary (necessary for real space PRG procedure) and returns a boolean.
        """
        M = np.asarray(M)
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
        old_variables : numpy array of float
            Time series data to be coarse grained, of shape (N, t), where N is the
            number of variables and t is the number of time samples.

        oldclu_idx : list of list of int
            Indices of the original variables composing each variable at the
            current RG step.

        cluster_method : str
            Similarity metric used to pair variables for coarse graining.
            Available options are:
            'pearson', 'spearman', 'mutual_information', 'cosine',
            'hamming', 'random'.

        Returns
        -------
        new_variables : numpy array of float
            Coarse-grained time series of shape (N/2, t), obtained by summing
            pairs of maximally correlated variables.

        corr_matrix : numpy array of float
            Similarity (correlation) matrix used during the pairing procedure.

        newclu_idx : list of list of int
            Updated indices of the original variables composing each
            coarse-grained variable.

        Notes
        -----
        Each variable is used exactly once per coarse-graining step.
        """
    #SELECT PAIRING METHOD
        if cluster_method == 'pearson':
            #Pearson
            corr_matrix = np.corrcoef(old_variables)
            #Spearman
        elif cluster_method == 'spearman':
            corr_matrix = spearmanr(old_variables, axis=1)[0]
            #Mutual Information
        elif cluster_method == 'mutual_information':
            corr_matrix = np.zeros((len(old_variables[:,0]),len(old_variables[:,0])))
            for i in range(len(old_variables[:,0])):
                for j in range(i, len(old_variables[:,0])):
                    if i != j:
                        mutual_info = mutual_info_score(old_variables[i], old_variables[j])
                        corr_matrix[i, j] = mutual_info
                        corr_matrix[j, i] = mutual_info  # Since MI is symmetric
            #Cosine distance
        elif cluster_method == 'cosine':
            corr_matrix = np.zeros((len(old_variables[:,0]),len(old_variables[:,0])))
            for i in range(len(old_variables[:,0])):
                for j in range(i, len(old_variables[:,0])):
                    if i != j:
                        cos_distance = distance.cosine(old_variables[i], old_variables[j])
                        corr_matrix[i, j] = 1 - cos_distance
                        corr_matrix[j, i] = 1 - cos_distance  # Since cosine distance is symmetric
            #Hamming distance
        elif cluster_method == 'hamming':
            corr_matrix = np.zeros((len(old_variables[:,0]),len(old_variables[:,0])))
            for i in range(len(old_variables[:,0])):
                for j in range(i, len(old_variables[:,0])):
                    if i != j:
                        hamming_distance = distance.hamming(old_variables[i], old_variables[j])
                        n = len(old_variables[i])
                        corr_matrix[i, j] = n - hamming_distance
                        corr_matrix[j, i] = n - hamming_distance  # Since hamming distance is symmetric

        elif cluster_method == 'random':
            corr_matrix = np.random.rand(len(old_variables[:,0]),len(old_variables[:,0]))
            corr_matrix += corr_matrix.T
            corr_matrix = corr_matrix/2

        N = len(corr_matrix[0])
        t = len(old_variables[0])

        halfN = int(N/2)
        new_variables = np.zeros((halfN,t), dtype=np.float64)
        newclu_idx = [[] for i in range(0,halfN)]

        for i in range(0,N):
            corr_matrix[i,i] = -np.inf

        return_matrix = corr_matrix.copy()
        for i in range(0,halfN):

            idx = np.unravel_index(np.argmax(corr_matrix),corr_matrix.shape) #returns a tuple
            new_variables[i,:] = old_variables[idx[0]] + old_variables[idx[1]] #sum ith row and jth row from argmax

            newclu_idx[i] = np.hstack((oldclu_idx[idx[0]],(oldclu_idx[idx[1]])))
            corr_matrix[idx[0]] = -np.inf
            corr_matrix[idx[1]] = -np.inf
            corr_matrix[:,idx[0]] = -np.inf
            corr_matrix[:,idx[1]] = -np.inf


        return new_variables, return_matrix, newclu_idx


    def get_CG_variables(self, binary_array, cluster_method="pearson", rg_steps=6):
        """
        Apply real-space PRG coarse graining to multidimensional binary time series.

        Parameters
        ----------
        binary_array : numpy array of int
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
        CG_var : list of numpy array of float
            Coarse-grained time series at each RG step.
            Entry k contains an array of shape (N_k, t), where N_k = N / 2^k.

        corr_matrix : list of numpy array of float
            Similarity (correlation) matrices used at each RG step.

        clu_idx : list of numpy array of int
            Indices of the original variables composing each coarse-grained
            variable at every RG step.
        """

        if self.is_binary_matrix(binary_array) == False:
            raise ValueError("Data time series should contain binarized values for real space PRG analysis. \nChoose one of the available binarizing methods (\"binary_from_stamps\", \"binary_zscore_threshold\", \"binary_zscore_maxima\") that best suits your data with the function binarize_data(data_array, bin_width=1, threshold=2, method= \"timestamps\")")
        if len(binary_array[:,0]) < 2**rg_steps:
            raise ValueError(f"Number of variables in data ({len(binary_array[:,0])}) is less than the requested coarse graining size ({2**rg_steps}). \nYou may reduce \"rg_steps\" to a suitable value.")
        if len(binary_array[:,0]) > len(binary_array[0,:]):
            warnings.warn(f"Number of variables in data ({len(binary_array[:,0])}) is greater than the number of samples ({len(binary_array[0,:])}) in a time window. This tend to yield unreliable and potentially spurious correlations between variables.")

        rg_steps+=1
        CG_var = [[] for i in range(0,rg_steps)]
        corr_matrix = [[] for i in range(0,rg_steps)]
        clu_idx = [[] for i in range(0,rg_steps)]

        CG_var[0] = binary_array.astype(np.float64)
        corr_matrix[0] = np.corrcoef(binary_array)
        clu_idx[0] =  np.arange((len(binary_array[:,0])))

        for i in range(1,rg_steps):
            CG_var[i], corr_matrix[i], clu_idx[i] = self.coarse_grain(CG_var[i-1],clu_idx[i-1],cluster_method)
        return CG_var, corr_matrix, clu_idx

