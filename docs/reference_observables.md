# Statistical Metrics (`observables.py`)

This module provides the methods necessary to track how statistical features and macroscopic observables evolve as the system undergoes recursive coarse graining. It analyzes a populated `CGVariables` object at each coarse graining step to evaluate specific scaling behaviors and calculate their corresponding power law exponents.

The module implements several measurements that can be used as signatures of scale-invariant dynamics, namely:

- **Mean Variance:** Tracks how variance scales with cluster size to determine if the scaling exponent falls between the bounds of independent and fully correlated limits.
- **Silence Probability:** Assesses the likelihood of a cluster remaining silent across scales, mapping its decay as a proxy of an effective free energy.
- **Covariance Spectrum:** Evaluates the full eigenvalue spectrum of the intracluster covariance matrix to check for scale-invariant shape collapse.
- **Max Covariance Eigenvalue:** Examines the scaling of the largest eigenvalue of the covariance matrix across scales.
- **Autocorrelation Function:** Computes the temporal autocorrelatoin of coarse grained variables across PRG iterations.
- **Decay Time:** Measures dynamical scaling by extracting the characteristic temporal decay from the autocorrelation function for progressively larger macroscopic clusters.
- **Activity Distribution:** Constructs probability distributions of the summed activity of coarse-grained variables. This allows the pipeline to determine whether the collective dynamics are converging toward a (trivial) Gaussian distribution or a non-Gaussian fixed point, also a signature of scale-invariant behavior.

::: prg_toolbox.observables
