import numpy as np
from sklearn.metrics import r2_score

def covariance_evals_and_evectors(x):
    """
    Calculates the data covariance matrix and diagonalizes it

    Parameters
    ----------
        x : ndarray of floats
            data as (x_i, t_j)

    Returns
    ----------
        evals : ndarray of floats 
            Vector of eigenvalues in decreasing order
        evectors : ndarray of floats
            Matrix of eigenvectors in decreasing order
    """
    x = x - np.mean(x,1,keepdims=True)
    cov_matrix = np.cov(x)
    evals, evectors = np.linalg.eigh(cov_matrix)
    index = np.flip(np.argsort(evals))
    evals = evals[index]
    evectors = evectors[:, index]
    return evals,evectors

def powerLaw_function(x,a,b):
    return a*np.power(x,b)

def get_scaling_exponent(CG_observable_values, spectrum = False, skip_first_value = False):
    """
    Estimate the scaling exponent of the PRG observables under coarse graining.

    Parameters
    ----------
    CG_observable_values : numpy array of float
        Observable values as a function of RG step k.

    spectrum : boolean
        When fitting the eigenvalue spectrum, the x-axis has 2**k elements (number of variables
        in a cluster) instead of k (number of rg_steps).
        Default is False. 

    skip_first_value : boolean
        Ignores initial value of the observable when it is trivial
        (e.g. there are no eigenvalues for 1x1 matrices in the 0-th step of coarse-graining).
        Default is False.

    Returns
    -------
    exponent : float
        Estimated power-law exponent.

    exponent_error : float
        One-sigma uncertainty of the exponent estimated from the fit covariance.

    exponent_r2 : float
        Coefficient of determination (R²) of the power-law fit.
    """

    x = CG_observable_values
    if skip_first_value:
        t = np.array([2**(i+1) for i in range(len(x))])
    elif not spectrum:
        t = np.array([2**i for i in range(len(x))])
    else:
        t = np.arange(len(x))+1

    log_t = np.log(t)
    log_x = np.log(x)

    fit_param, fit_cov = np.polyfit(log_t, log_x, 1, cov=True)

    exponent_error = np.sqrt(fit_cov[0,0])
    exponent = fit_param[0]
    constant = fit_param[1]
    log_x_pred = exponent * log_t + constant
    exponent_r2 = r2_score(log_x, log_x_pred)

    return exponent, exponent_error, exponent_r2

