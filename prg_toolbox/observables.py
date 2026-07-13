"""
Copyright (c) 2026 Daniel Miranda Castro. Licensed under the MIT License.

Statistical Observables and Scaling Analysis for Real-Space PRG.

This module implements the measurements (observables) used to analyse 
scale invariance signatures across coarse-grained variables.

Each class takes a populated `CGVariables` object and calculates its statistical properties as 
a function of the coarse-graining step $k$, automatically extracting scaling 
exponents via power-law least-squares fitting.

Calculated observables include:

    * `mean_variance`: Scaling of cluster activity mean variance.
    * `log_silence_probability`: Scaling of probability of having no activity
        within a cluster (free-energy proxy).
    * `max_covariance_eigenvalue`: Scaling of the dominant covariance eigenvalue.
    * `covariance_spectrum`: Intracluster eigenvalue spectrum scaling alongside 
      Marchenko-Pastur random matrix null models.
    * `autocorrelation_function` & `decay_time`: Scaling of autocorrelation decay
        (critical slowing down).
    * `activity_distribution`: Probability density profile across transformations.
"""
import numpy as np
import warnings
from scipy.stats import moment
from scipy.optimize import curve_fit
# You mentioned these exist in a utils file elsewhere:
from .utils import (covariance_evals_and_evectors, 
                    get_scaling_exponent)
# Import the class for type checking
from .coarse_graining import CGVariables

class mean_variance:
    def __init__(self, CG_variables):
        """
        Compute and analyze the scaling of mean variance under real-space PRG.

        This class extracts the mean variance of coarse-grained variables
        obtained from a `CG_variables` object and estimates the
        corresponding scaling exponent.

        Parameters
        ----------
        CG_variables : CG_variables output object from the real-space
            coarse graining procedure, containing coarse-grained time series and
            metadata.

        Attributes
        ----------
        time_window : int
            Temporal lengths parsed directly from the data wrapper.
        rg_steps : int
            Number of coarse graining iterations.
        values : ndarray
            Calculated average variance evaluated across steps.
        exponent : float
            Estimated scaling exponent (alpha).
        exponent_error : float
            One-sigma fitting uncertainty for the scaling exponent.
        exponent_r2 : float
            Goodness of fit metric ($R^2$).
        avg_across_windows : ndarray
            Used to simplify handling of multi-trial data (average across trials)
        std_across_windows : ndarray
            Used to simplify handling of multi-trial data (standard deviation across trials)
        """

        if type(CG_variables) != CGVariables:
            raise ValueError("Input must be of type CG_variables.\n Feed your data into the CG_variables method and use its output in this function.")

        self.time_window = CG_variables.time_window

        rg_steps = len(CG_variables.CG_timeseries)
        self.rg_steps = rg_steps-1

        values = self.get_mean_variance(CG_variables.CG_timeseries, rg_steps)
        self.values = values
        alpha, alpha_error, alpha_r2 = get_scaling_exponent(values)
        self.exponent = alpha
        self.exponent_error = alpha_error
        self.exponent_r2 = alpha_r2

        #to simplify handling of multi-trial data 
        self.avg_across_windows = self.values
        self.std_across_windows = np.zeros(len(self.values))

    def get_mean_variance(self, CG_timeseries, rg_steps):
        """
        Compute the mean variance of coarse-grained variables at each PRG iteration.

        For a given renormalization group (RG) step k, the variance of each
        coarse-grained variable is computed across time, and the mean over
        all variables at that RG step is returned.
        """
        mean_variance = np.zeros((rg_steps))
        for k in range(0,rg_steps):
            for j in range(0,len(CG_timeseries[k])):
                mean_variance[k] += np.var(CG_timeseries[k][j])
            mean_variance[k] = mean_variance[k]/len(CG_timeseries[k])

        return mean_variance

class log_silence_probability:
    def __init__(self, CG_variables):
        """
        Compute the scaling of log silence probability under real-space PRG.

        Measures the probability that a block variable is entirely quiet.
        
        Parameters
        ----------
        CG_variables :  CG_variables output object from the real-space
            coarse graining procedure, containing coarse-grained time series and
            metadata.

        Attributes
        ----------
        time_window : int
            Temporal lengths parsed directly from the data wrapper.
        rg_steps : int
            Number of coarse graining iterations.
        values : ndarray
            Negative log probability values across steps.
        exponent : float
            Estimated scaling exponent (beta).
        exponent_error : float
            One-sigma fitting uncertainty for the scaling exponent.
        exponent_r2 : float
            Goodness of fit metric ($R^2$).
        avg_across_windows : ndarray
            Used to simplify handling of multi-trial data (average across trials)
        std_across_windows : ndarray
            Used to simplify handling of multi-trial data (standard deviation across trials)
        """

        if type(CG_variables) != CGVariables:
            raise ValueError("Input must be of type CG_variables.\n Feed your data into the CG_variables method and use its output in this function.")

        self.time_window = CG_variables.time_window

        rg_steps = len(CG_variables.CG_timeseries)
        self.rg_steps = rg_steps-1

        values = self.get_log_silence_probability(CG_variables.CG_timeseries, rg_steps)
        self.values = values
        beta, beta_error, beta_r2 = get_scaling_exponent(values)
        self.exponent = beta
        self.exponent_error = beta_error
        self.exponent_r2 = beta_r2

        #to simplify handling of multi-trial data 
        self.avg_across_windows = self.values
        self.std_across_windows = np.zeros(len(self.values))

        # A step's silence probability is clipped to 1e-6 (-> -log(1e-6)) when
        # no coarse-grained variable was ever silent at that step.
        clipped = np.any(np.isclose(values, -np.log(0.000001)))

        if clipped:
            warnings.warn(f" Coarse-graining yielded variables with no silence. Consider changing preprocessing or PRG parameters.")

    def get_log_silence_probability(self, CG_timeseries, rg_steps):
        """
        Compute the log silence probability of coarse-grained variables at each PRG iteration.

        For a given renormalization group (RG) step k, the fraction of time bins where a coarse-grained variable
        is completely silent is computed, and the mean over
        all variables at that RG step is returned.
        """
        silence_probability = np.zeros((rg_steps))
        for i in range(0,rg_steps):
            n_silent = 0
            for j in range(0,len(CG_timeseries[i][:,0])):
                #count silent steps and divide by time series length: (nSteps-nSpikes)/nSteps
                n_silent += 1 - np.count_nonzero(CG_timeseries[i][j])/len(CG_timeseries[i][j])
            #divide by number of neurons
            n_silent /= len(CG_timeseries[i][:,0])
            if n_silent == 0:
                silence_probability[i] = 0.000001
            else:
                silence_probability[i] = n_silent
        return -np.log(silence_probability)

class max_covariance_eigenvalue:
    def __init__(self, CG_variables):
        """
        Analyze the scaling of the maximum covariance eigenvalue under real-space PRG.

        Tracks the dominant shared mode variance trend within emergent cluster blocks.

        Parameters
        ----------
        CG_variables :  CG_variables output object from the real-space
            coarse graining procedure, containing coarse-grained time series and
            metadata.

        Attributes
        ----------
        time_window : int
            Temporal lengths parsed directly from the data wrapper.
        rg_steps : int
            Number of coarse graining iterations.
        values : ndarray
            Averaged maximal eigenvalues computed across steps.
        exponent : float
            Calculated scaling exponent (epsilon).
        exponent_error : float
            One-sigma fitting uncertainty for the scaling exponent.
        exponent_r2 : float
            Goodness of fit metric ($R^2$).
        avg_across_windows : ndarray
            Used to simplify handling of multi-trial data (average across trials)
        std_across_windows : ndarray
            Used to simplify handling of multi-trial data (standard deviation across trials)
        """

        if type(CG_variables) != CGVariables:
            raise ValueError("Input must be of type CG_variables.\n Feed your data into the CG_variables method and use its output in this function.")

        self.time_window = CG_variables.time_window

        rg_steps = len(CG_variables.CG_timeseries)
        self.rg_steps = rg_steps-1

        values = self.get_max_covariance_eigenvalue(CG_variables.CG_timeseries, CG_variables.CG_cluster_idx, rg_steps)
        self.values = values
        epsilon, epsilon_error, epsilon_r2 = get_scaling_exponent(values[1:],skip_first_value = True)
        self.exponent = epsilon
        self.exponent_error = epsilon_error
        self.exponent_r2 = epsilon_r2

        #to simplify plotting script
        self.avg_across_windows = self.values
        self.std_across_windows = np.zeros(len(self.values))


    def get_max_covariance_eigenvalue(self, CG_timeseries, CG_cluster_idx, rg_steps):
        """
        Compute the covariance spectrum of coarse-grained variables at each PRG iteration.

        For a given renormalization group (RG) step k, the largest (intracluster) covariance eigenvalue of each
        coarse-grained variable is computed, and the mean over
        all variables (clusters) at that RG step is returned.
        """
        N = len(CG_timeseries[0][:,0])
        max_eval = np.zeros((rg_steps))
        original_timeseries = CG_timeseries[0]
        for i in range (1,rg_steps):
            for j in range(0, int(N/2**i)):
                one_cluster_max_eval = covariance_evals_and_evectors(original_timeseries[CG_cluster_idx[i][j]])[0][0]
                max_eval[i] += one_cluster_max_eval
            max_eval[i] *= 2**i/N
        return max_eval

class covariance_spectrum:
    def __init__(self, CG_variables, spectrum_fit_length = 1/5):
        """
        Analyze the scaling of the intracluster covariance spectra patterns under PRG.

        Fits power-law trends to rank-ordered spectrum curves and compares 
        the bulk eigenvalue density with Marchenko-Pastur theoretical limits.

        Parameters
        ----------
        CG_variables :  CG_variables output object from the real-space
            coarse graining procedure, containing coarse-grained time series and
            metadata.
        spectrum_fit_length : float, optional
            Fractional cutoff specifying how much of the tail rank spectrum to use 
            to estimate the spectral exponent mu. Default is 1/5.

        Attributes
        ----------
        time_window : int
            Temporal lengths parsed directly from the data wrapper.
        rg_steps : int
            Number of coarse graining iterations.
        fit_length : int
            spectrum_fit_length argument parsed directly from the arguments.
        values : list of ndarray
            Rank-ordered eigenvalue collections evaluated for each scaling step $k$.
        exponent : float
            Calculated scaling exponent (mu).
        mp_x_fit : ndarray
            X-coordinates evaluating Marchenko-Pastur null curves.
        mp_y_fit : ndarray
            Theoretical probability density function values for the MP distribution.
        mp_lambda_plus : float
            Analytical upper edge noise bound predicted by Marchenko-Pastur bulk fits.
        pdf_pl_exponent : float
            Power-law exponent obtained from Maximum Likelihood Estimation (MLE) of
            bulk eigenvalues density, which can be analitically compared to exponent mu.
        """

        if type(CG_variables) != CGVariables:
            raise ValueError("Input must be of type CG_variables.\n Feed your data into the CG_variables method and use its output in this function.")

        self.time_window = CG_variables.time_window

        rg_steps = len(CG_variables.CG_timeseries)
        self.rg_steps = rg_steps-1
        self.fit_length = int(spectrum_fit_length*2**(self.rg_steps))

        values = self.get_covariance_spectrum(CG_variables.CG_timeseries, CG_variables.CG_cluster_idx, rg_steps)
        self.values = values
        mu, mu_error, mu_r2 = get_scaling_exponent(values[-1][1:self.fit_length], spectrum=True)
        self.exponent = -mu
        self.exponent_error = mu_error
        self.exponent_r2 = mu_r2

        ##to simplify handling of multi-trial data 
        self.avg_across_windows = self.values
        self.std_across_windows = [np.zeros_like(self.values[k]) for k in range(len(self.values))]

        # calculate Marchenko-Pastur null hypothesis
        self.mp_x_fit, self.mp_y_fit, self.mp_lambda_plus, self.mp_sigma, self.mp_Q = self.get_marchenko_pastur(CG_variables.CG_timeseries[0])
        # fit power law to probability distribution
        self.pdf_pl_exponent, self.pdf_pl_normalization_constant = self.get_distribution_power_law()


    def get_covariance_spectrum(self, CG_timeseries, CG_cluster_idx, rg_steps):
        """
        Compute the covariance spectrum of coarse-grained variables at each PRG iteration.

        For a given renormalization group (RG) step k, the (intracluster) covariance eigenvalues of each
        coarse-grained variable is computed, and the mean over
        all variables (clusters) at that RG step is returned.
        """

        N = len(CG_timeseries[0][:,0])
        original_timeseries = CG_timeseries[0]
        evals_k = [np.zeros((2**i)) for i in range(0,rg_steps)]
        for i in range (1,rg_steps):
            for j in range(0, int(N/2**i)):
                one_eval_set = covariance_evals_and_evectors(original_timeseries[CG_cluster_idx[i][j]])[0]
                evals_k[i] += one_eval_set
            evals_k[i] *= 2**i/N
        return evals_k
    
    def marchenko_pastur_pdf(self, x, Q, sigma_sq):
        """
        Computes the theoretical Marchenko-Pastur PDF.
        """
        lambda_minus = sigma_sq * (1 - np.sqrt(1/Q))**2
        lambda_plus  = sigma_sq * (1 + np.sqrt(1/Q))**2
        
        y = np.zeros_like(x)
        valid = (x >= lambda_minus) & (x <= lambda_plus)
        if np.any(valid):
            y[valid] = (Q / (2 * np.pi * sigma_sq)) * \
                    (np.sqrt((lambda_plus - x[valid]) * (x[valid] - lambda_minus)) / x[valid])
        return y   
    
    def get_marchenko_pastur(self, raw_timeseries):
        """
        Computes and fits the Marchenko-Pastur distribution to the covariance 
        eigenvalues of the final RG step. Populates the object with MP attributes.
        """
        # Grab the eigenvalues of the final RG step
        eigenvalues = self.avg_across_windows[-1] 
        
        N = len(eigenvalues)
        T = raw_timeseries.shape[1]
        mp_Q = T / N 

        # Fit a subset to avoid letting massive signal outliers skew the noise fit
        bulk_eigenvalues = eigenvalues[eigenvalues < np.percentile(eigenvalues, 95)]
        counts, bin_edges = np.histogram(bulk_eigenvalues, bins=60, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        def fit_func(x, sigma_sq):
            return self.marchenko_pastur_pdf(x, mp_Q, sigma_sq)

        # Perform the curve fit
        initial_guess = [np.mean(bulk_eigenvalues)]
        try:
            popt, _ = curve_fit(fit_func, bin_centers, counts, p0=initial_guess, bounds=(1e-5, np.inf), maxfev=5000)
            mp_sigma = popt[0]
        except RuntimeError as e:
            print(f"Warning: MP fit failed to converge. Falling back to analytical mean estimate.")
            # Theoretical mean of MP is sigma_sq, so the mean of the bulk is an excellent proxy
            mp_sigma = np.mean(bulk_eigenvalues) 
        
        # Store all calculated data as object attributes for the plotting script
        mp_lambda_plus = mp_sigma * (1 + np.sqrt(1/mp_Q))**2
        mp_x_fit = np.linspace(0, np.max(eigenvalues), 1000)
        mp_y_fit = self.marchenko_pastur_pdf(mp_x_fit, mp_Q, mp_sigma)
        
        return mp_x_fit, mp_y_fit, mp_lambda_plus, mp_sigma, mp_Q
 
    def get_distribution_power_law(self, exclude_max_eval=True):
        """
        Fits a power law directly to the empirical probability distribution 
        of the eigenvalues in the tail using Maximum Likelihood Estimation (MLE).
        Returns the exponent, x coordinates, and y coordinates for plotting.
        """
        # 1. Filter for eigenvalues strictly in the tail
        eigenvalues = self.avg_across_windows[-1]
        tail_start = self.mp_lambda_plus
        tail_evals = eigenvalues[eigenvalues > tail_start]
        # Exclude the maximum eigenvalue, which is often an extreme outlier 
        # related to global fluctuations and can distort the fit.
        if exclude_max_eval:
            tail_evals = tail_evals[1:]  
        n_tail = len(tail_evals)
        n_total = len(eigenvalues)
        
        # Safety check: ensure we have enough points in the tail to fit
        if n_tail < 3:
            return np.nan, np.nan
            
        # 2. Maximum Likelihood Estimation for the continuous power law exponent
        # Ref.: Clauset, Shalizi, and Newman (2009) DOI. 10.1137/070710111
        pdf_exponent = 1 + n_tail / np.sum(np.log(tail_evals / tail_start))

        # 3. Scale the PDF to match the density=True histogram
        # A pure PDF integrates to 1. However, our histogram's tail only represents 
        # a fraction of the total area. We must scale the curve by that fraction.
        fraction_in_tail = n_tail / n_total
        
        # The theoretical constant C for p(x) = C * (x/xmin)^-alpha
        normalization_constant = fraction_in_tail * (pdf_exponent - 1) / tail_start
        
        return pdf_exponent, normalization_constant

class _nth_moment:
    # Not currently included in the __init__.py import list,
    def __init__(self, CG_variables, order=4):
        """
        Compute and analyze the scaling of n-th statistical moment under real-space PRG.

        This class extracts the n-th statistical moment of coarse-grained variables
        obtained from a `CG_variables` object and estimates the
        corresponding scaling exponent.

        Parameters
        ----------
        CG_variables :  CG_variables output object from the real-space
            coarse graining procedure, containing coarse-grained time series and
            metadata.

        Attributes
        ----------
        time_window : int
            Temporal lengths parsed directly from the data wrapper.
        rg_steps : int
            Number of coarse graining iterations.
        values : ndarray
            Calculated average nth-moment evaluated across steps.
        exponent : float
            Calculated scaling exponent (alpha_n).
        exponent_error : float
            One-sigma fitting uncertainty for the scaling exponent.
        exponent_r2 : float
            Goodness of fit metric ($R^2$).
        avg_across_windows : ndarray
            Used to simplify handling of multi-trial data (average across trials)
        std_across_windows : ndarray
            Used to simplify handling of multi-trial data (standard deviation across trials)
        """

        if type(CG_variables) != CGVariables:
            raise ValueError("Input must be of type CG_variables.\n Feed your data into the CG_variables method and use its output in this function.")

        self.time_window = CG_variables.time_window
        self.order = order

        rg_steps = len(CG_variables.CG_timeseries)
        self.rg_steps = rg_steps-1

        values = self.get_nth_moment(CG_variables.CG_timeseries, rg_steps, order)
        self.values = values
        alpha_n, alpha_n_error, alpha_n_r2 = get_scaling_exponent(values)
        self.exponent = alpha_n
        self.exponent_error = alpha_n_error
        self.exponent_r2 = alpha_n_r2

        #to simplify handling of multi-trial data 
        self.avg_across_windows = self.values
        self.std_across_windows = np.zeros(len(self.values))


    def get_nth_moment(self, CG_timeseries, rg_steps, order=4):
        """
        Compute the nth statistical moment of coarse-grained variables at each PRG iteration.

        For a given renormalization group (RG) step k, the moment of each
        coarse-grained variable is computed across time, and the mean over
        all variables at that RG step is returned.
        """
        nth_moment = np.zeros((rg_steps))
        for k in range(0,rg_steps):
            for j in range(0,len(CG_timeseries[k])):
                nth_moment[k] += moment(CG_timeseries[k][j], order=order)
            nth_moment[k] = nth_moment[k]/len(CG_timeseries[k])

        return nth_moment

class autocorrelation_function:
    def __init__(self, CG_variables):
        """
        Compute and analyze the scaling of zero-lag autocorrelation functions under real-space PRG.

        This class extracts the mean autocorrelation of coarse-grained variables
        obtained from a `CG_variables` object. 
        
        Note: this observable does not estimate a scaling exponent, although it 
        is used in the `decay_time` class to estimate the characteristic decay time 
        scaling exponent.

        Parameters
        ----------
        CG_variables : CG_variables output object from the real-space
            coarse graining procedure, containing coarse-grained time series and
            metadata.

        Attributes
        ----------
        time_window : int
            Temporal lengths parsed directly from the data wrapper.
        rg_steps : int
            Number of coarse graining iterations.
        values : ndarray
            Autocorrelation function evaluated for each scaling step $k$..

        across_windows : ndarray
            Used to simplify handling of multi-trial data (average across trials)
        std_across_windows : ndarray
            Used to simplify handling of multi-trial data (standard deviation across trials)
        """

        if type(CG_variables) != CGVariables:
            raise ValueError("Input must be of type CG_variables.\n Feed your data into the CG_variables method and use its output in this function.")

        self.time_window = CG_variables.time_window

        rg_steps = len(CG_variables.CG_timeseries)
        self.rg_steps = rg_steps-1

        values = self.get_autocorrelation_function(CG_variables.CG_timeseries, rg_steps)
        self.values = values

        #to simplify handling of multi-trial data 
        self.avg_across_windows = self.values
        self.std_across_windows = np.zeros_like(self.values)


    def get_autocorrelation_function(self, CG_timeseries, rg_steps):
        """
        Compute the mean normalized autocorrelation function of coarse-grained
        variables at each PRG iteration.

        For each renormalization group (RG) step k, this function computes the
        autocorrelation function of every coarse-grained variable after
        demeaning its time series. The autocorrelations are averaged across
        variables and normalized such that the zero-lag value equals one.

        The result is a function-valued observable: an autocorrelation curve
        as a function of time lag for each RG step.

        Parameters
        ----------
        CG_timeseries : list of ndarray
            Coarse-grained time series data across RG steps.
            Entry k is a 2D array of shape (N_k, T), where N_k is the number
            of coarse-grained variables at RG step k and T is the number
            of time samples per variable.

        rg_steps : int
            Number of renormalization group (coarse-graining) steps.

        Returns
        -------
        mean_autocorrelation : list of numpy.ndarray
            List of length ``rg_steps``. Entry k is a 1D ndarray of length
            ``2*T - 1`` representing the mean autocorrelation function at RG
            step k, averaged across variables and normalized by its zero-lag
            value.
        """
        mean_autocorrelation = [np.zeros((2*len(CG_timeseries[i][0])-1)) for i in range(rg_steps)]
        for k in range(rg_steps):
            N = len(CG_timeseries[k][:,0])
            for j in range(N):
                demeaned_data = CG_timeseries[k][j] - np.mean(CG_timeseries[k][j])
                this_autocorrelation = np.correlate(demeaned_data, demeaned_data, mode='full')
                mean_autocorrelation[k] += this_autocorrelation
            mean_autocorrelation[k] = mean_autocorrelation[k]/N
            mean_autocorrelation[k] = mean_autocorrelation[k]/mean_autocorrelation[k][np.argmax(mean_autocorrelation[k])]

        return mean_autocorrelation

class decay_time:
    def __init__(self, CG_variables, correlation_bins=10):
        """
        Compute and analyze the scaling of the autocorrelation decay time
        under real-space PRG.

        Extracts the characteristic exponential time from autocorrelation
        functions of coarse-grained variables and estimates the corresponding scaling exponent.

        Parameters
        ----------
        CG_variables : CG_variables output object from the real-space
            coarse graining procedure, containing coarse-grained time series and
            metadata.


        correlation_bins : int, optional
            Number of time bins used to compute the autocorrelation decay.
            Default is 10.

        Attributes
        ----------
        time_window : int
            Temporal lengths parsed directly from the data wrapper.
        rg_steps : int
            Number of coarse graining iterations.
        values : ndarray
            Calculated characteristic decay time evaluated across steps.
        exponent : float
            Estimated scaling exponent (z).
        exponent_error : float
            One-sigma fitting uncertainty for the scaling exponent.
        exponent_r2 : float
            Goodness of fit metric ($R^2$).
        avg_across_windows : ndarray
            Used to simplify handling of multi-trial data (average across trials)
        std_across_windows : ndarray
            Used to simplify handling of multi-trial data (standard deviation across trials)
        """

        if type(CG_variables) != CGVariables:
            raise ValueError("Input must be of type CG_variables.\n Feed your data into the CG_variables method and use its output in this function.")


        # Generate the required intermediate object
        ac_function = autocorrelation_function(CG_variables)

        self.time_window = ac_function.time_window

        # ac_function.rg_steps is already decremented once (it excludes the
        # trivial step-0 count); use the full step count here so `values`
        # covers every RG step, consistent with every other observable class.
        rg_steps = len(ac_function.avg_across_windows)
        self.rg_steps = rg_steps-1

        values = self.get_decay_time(ac_function.avg_across_windows, correlation_bins, rg_steps)
        self.values = values
        z, z_error, z_r2 = get_scaling_exponent(values)
        self.exponent = z
        self.exponent_error = z_error
        self.exponent_r2 = z_r2

        #to simplify plotting script
        self.avg_across_windows = self.values
        self.std_across_windows = np.zeros(len(self.values))

    def _exponential_function(self, x,a,b,c):
        return a*np.exp(-x/b) + c

    def get_decay_time(self, ac_values, nbins, rg_steps):
        """
        Compute the characteristic decay time of coarse-grained variables autocorrelation
        function (ac_values) at each PRG iteration.

        For a given renormalization group (RG) step k, the autocorrelation exponential
        decay time is computed taking into account the first ``nbins`` values. 
        """
        first_bin = int((len(ac_values[0])-1)/2)
        decay_time = np.zeros(rg_steps)

        x_data = np.arange(nbins)

        for k in range(rg_steps):
            # Slice the y-data
            y_data = ac_values[k][first_bin : (first_bin + nbins)]

            # Handle non-positive values to avoid log errors
            mask = y_data > 0
            if np.any(mask):
                log_y = np.log(y_data[mask])
                x_masked = x_data[mask]

                # Fit to a 1st degree polynomial: log(y) = slope*x + intercept
                # slope is params[0], intercept is params[1]
                params = np.polyfit(x_masked, log_y, 1)
                
                slope = params[0]
                
                # tau = -1 / slope. 
                # We use a small epsilon or check to avoid division by zero
                if slope < 0:
                    decay_time[k] = -1.0 / slope
                else:
                    decay_time[k] = 0 #np.nan
            else:
                decay_time[k] = 0 #np.nan

        return decay_time

class activity_distribution:
    def __init__(self, CG_variables):
        """
        Compute and analyze the scaling of coarse-grained variables activity distribution under real-space PRG.

        Extracts the probability density of (summed) activity of coarse-grained variables.

        Parameters
        ----------
        CG_variables : CG_variables output object from the real-space
            coarse graining procedure, containing coarse-grained time series and
            metadata.

        Attributes
        ----------
        time_window : int
            Temporal lengths parsed directly from the data wrapper.
        rg_steps : int
            Number of coarse graining iterations.
        values : ndarray
            Summed activity probability density function evaluated for each scaling step $k$..

        across_windows : ndarray
            Used to simplify handling of multi-trial data (average across trials)
        std_across_windows : ndarray
            Used to simplify handling of multi-trial data (standard deviation across trials)
        """

        if type(CG_variables) != CGVariables:
            raise ValueError("Input must be of type CG_variables.\n Feed your data into the CG_variables method and use its output in this function.")

        self.time_window = CG_variables.time_window

        rg_steps = len(CG_variables.CG_timeseries)
        self.rg_steps = rg_steps-1

        values = self.get_activity_distribution(CG_variables.CG_timeseries, rg_steps)
        self.values = values

        #to simplify plotting script
        self.avg_across_windows = self.values
        self.std_across_windows = [np.zeros_like(self.values[k]) for k in range(len(self.values))]

        #x axis
        self.normalized_activity = [np.arange((1+2**i))/2**i for i in range(rg_steps)]

    def get_activity_distribution(self, CG_timeseries, rg_steps):
        """
        Compute the activity distribution of coarse-grained
        variables at each PRG iteration.

        For each renormalization group (RG) step k, this function computes the
        probability density for the summed activity of all coarse-grained variables.
        The densities are averaged across variables and normalized such that the
        maximum activity at each step equals one.

        The result is a function-valued observable: a probability density curve
        for each RG step.

        Parameters
        ----------
        CG_timeseries : list of numpy.ndarray
            Coarse-grained time series data across RG steps.
            Entry k is a 2D array of shape (N_k, T), where N_k is the number
            of coarse-grained variables at RG step k and T is the number
            of time samples per variable.

        rg_steps : int
            Number of renormalization group (coarse-graining) steps.

        Returns
        -------
        probability_density : list of numpy.ndarray
            List of length ``rg_steps``. Entry k is a 1D ndarray of length
            ``k**2`` representing the normalized activity distribution at RG
            step k.
        normalized_activity : list of numpy.ndarray
            List of length ``rg_steps``. Entry k is a 1D ndarray of length
            ``k**2`` representing the normalized x-axis (possible discrete values
            for the summed activity).
        """

        probability_density = [np.zeros((1+2**i)) for i in range(rg_steps)]                  #y axis
        for i in range(rg_steps):
            nvariables = len(CG_timeseries[i])
            nvalues = 1+2**i
            counter = np.zeros((nvariables, nvalues))
            for l in range(nvalues):
                for j in range(nvariables):
                    counter[j,l] = np.count_nonzero(CG_timeseries[i][j,:] == l)

            probability_density[i] = np.mean(counter, axis=0)
            dx = 1/(2**i)
            probability_density[i] /= np.sum(probability_density[i])*dx

        return probability_density


class _avalanche_covariance_eigenvalue:
    # Not currently included in the __init__.py import list, 
    # but this is where it would go if we wanted to add it as an observable.
    def __init__(self, CG_variables):
        """
        Analyze the scaling of the `avalanche` covariance eigenvalue under real-space PRG.

        Computes the dominant covariance eigenvalue out of the index-shuffled raw time series
        (summed activity at any given time step remains the same). Then, tracks the trend within 
        emergent cluster blocks.

        Parameters
        ----------
        CG_variables :  CG_variables output object from the real-space
            coarse graining procedure, containing coarse-grained time series and
            metadata.

        Attributes
        ----------
        time_window : int
            Temporal lengths parsed directly from the data wrapper.
        rg_steps : int
            Number of coarse graining iterations.
        values : ndarray
            Averaged maximal eigenvalues computed across steps.
        exponent : float
            Calculated scaling exponent (epsilon_avalanche).
        exponent_error : float
            One-sigma fitting uncertainty for the scaling exponent.
        exponent_r2 : float
            Goodness of fit metric ($R^2$).
        avg_across_windows : ndarray
            Used to simplify handling of multi-trial data (average across trials)
        std_across_windows : ndarray
            Used to simplify handling of multi-trial data (standard deviation across trials)
        """

        if type(CG_variables) != CGVariables:
            raise ValueError("Input must be of type CG_variables.\n Feed your data into the CG_variables method and use its output in this function.")

        self.time_window = CG_variables.time_window

        shuffled_timeseries = CG_variables.CG_timeseries[0].copy()
        rng = np.random.default_rng()
        rng.permuted(shuffled_timeseries, axis=0, out=shuffled_timeseries)
        # run coarse graining again on the shuffled data
        # this double calculates coarse-graining, but makes all classes take the same input
        shuffled_CG_variables = CGVariables(shuffled_timeseries, cluster_method=CG_variables.cluster_method, rg_steps=CG_variables.rg_steps)

        rg_steps = len(CG_variables.CG_timeseries)
        self.rg_steps = rg_steps-1

        values = self.get_max_covariance_eigenvalue(shuffled_CG_variables.CG_timeseries, shuffled_CG_variables.CG_cluster_idx, rg_steps)
        self.values = values
        epsilon, epsilon_error, epsilon_r2 = get_scaling_exponent(values[1:],skip_first_value = True)
        self.exponent = epsilon
        self.exponent_error = epsilon_error
        self.exponent_r2 = epsilon_r2

        #to simplify plotting script
        self.avg_across_windows = self.values
        self.std_across_windows = np.zeros(len(self.values))


    def get_max_covariance_eigenvalue(self, CG_timeseries, CG_cluster_idx, rg_steps):
        """
        Compute the covariance spectrum of coarse-grained variables at each PRG iteration.

        For a given renormalization group (RG) step k, the largest (intracluster) covariance eigenvalue of each
        coarse-grained variable is computed, and the mean over
        all variables (clusters) at that RG step is returned.
        """
        N = len(CG_timeseries[0][:,0])
        max_eval = np.zeros((rg_steps))
        original_timeseries = CG_timeseries[0]
        for i in range (1,rg_steps):
            for j in range(0, int(N/2**i)):
                one_cluster_max_eval = covariance_evals_and_evectors(original_timeseries[CG_cluster_idx[i][j]])[0][0]
                max_eval[i] += one_cluster_max_eval
            max_eval[i] *= 2**i/N
        return max_eval