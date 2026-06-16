print('am I here')
from .plot_mean_variance import plot_mean_variance
print('after mv')
from .plot_log_silence_probability import plot_log_silence_probability
print('after sp')
from .plot_max_covariance_eigenvalue import plot_max_covariance_eigenvalue

from .plot_covariance_spectrum import plot_covariance_spectrum
print('after spec')
from .plot_autocorrelation_function import plot_autocorrelation_function
from .plot_decay_time import plot_decay_time
from .plot_activity_distribution import plot_activity_distribution
print('after a lot')
from .plot_avalanche_covariance_eigenvalue import plot_avalanche_covariance_eigenvalue
print('ha')

# Optional: explicitly define what is available publicly in this submodule
__all__ = [
    "plot_mean_variance",
    "plot_log_silence_probability",
    "plot_max_covariance_eigenvalue",
    "plot_covariance_spectrum",
    "plot_autocorrelation_function",
    "plot_decay_time",
    "plot_activity_distribution",
    "plot_avalanche_covariance_eigenvalue"
]