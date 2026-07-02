import numpy as np

def init_firing_probabilities(Nsamples):
    return np.random.rand(Nsamples)

def binomial_process_single(Nsamples, p_firing):
    mask = np.random.rand(Nsamples) < p_firing
    return mask * 1

def binomial_process(Nvar, Nsamples, p_firing):
    x_timeseries = np.zeros((Nsamples, Nvar))
    for idx in range(Nvar):
        x_timeseries[:, idx] = binomial_process_single(Nsamples, p_firing)

    return x_timeseries

"""
-------------
If you don't have numba or you don't want to use it
set the following flag to False.
Numba allows for (much) faster execution and in this
case for parallel computation as well.
-------------
"""

use_numba = True
if use_numba:
    from numba import njit
    from numba import prange

    @njit
    def binomial_process_single(Nsamples, p_firing):
        mask = np.random.rand(Nsamples) < p_firing
        return mask * 1

    @njit(parallel = True)
    def binomial_process(Nvar, Nsamples, p_firing):
        x_timeseries = np.zeros((Nsamples, Nvar))
        for idx in prange(Nvar):
            x_timeseries[:, idx] = binomial_process_single(Nsamples, p_firing)

        return x_timeseries
