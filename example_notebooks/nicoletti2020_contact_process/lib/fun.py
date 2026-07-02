import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def evaluate_cov_corr(x_timeseries, plot_diagonal = True):
    if x_timeseries.shape[0] < x_timeseries.shape[1]:
        print('The number of samples is smaller than the number of variables.')
        print('Check the shape of the input array!')

    print('Building covariance matrix...')
    x_timeseries = np.array(x_timeseries)
    covariance = np.cov(x_timeseries, rowvar = False)
    correlations = np.corrcoef(x_timeseries.T)
    print('Done')

    if np.argwhere(np.isnan(correlations)).size > 0:
        print('Some sites have been silent for all the run. Some diagonal elements are zero!')

    print('Printing correlation matrix...')
    correlations_nodiagonal = correlations.copy()
    if not plot_diagonal:
        np.fill_diagonal(correlations_nodiagonal, 0)

    elev_min = correlations_nodiagonal.min()
    elev_max = correlations_nodiagonal.max()
    mid_val = 0

    fig, ax = plt.subplots(figsize = (8,7))
    c = ax.imshow(correlations_nodiagonal, cmap = 'RdBu_r',
                  clim=(elev_min, elev_max),
                  norm=MidpointNormalize(midpoint=mid_val,vmin=elev_min, vmax=elev_max),
                  interpolation = 'nearest',
                  aspect = 'auto')
    fig.colorbar(c, ax=ax)
    plt.xticks([])
    plt.yticks([])
    plt.show()

    return covariance, correlations
