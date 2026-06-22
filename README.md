
# Phenomenological Renormalization Group (PRG) Toolbox

[![Documentation](https://img.shields.io/badge/docs-live-brightgreen.svg)](https://damiaom.github.io/prg_toolbox/)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


PRG Toolbox is a Python package that implements the Phenomenological Renormalization Group (PRG) coarse graining method introduced by Meshulam et al. (2019) [1,2]. It performs model-free analysis designed to study scale-invariant behavior and collective dynamics in high-dimensional multivariate time series, such as multi-neuron spike recordings or rs-fMRI data [3,4].

## Method Overview

The toolbox applies the intuition of the renormalization group from statistical physics directly to empirical data without relying on an explicit underlying model. Near a second-order phase transition, the collective dynamics of a complex system become insensitive to most microscopic details, displaying scale-invariant activity. 

The method begins with $N$ individual binary variables, which the algorithm clusters into pairs or blocks. For each block, the underlying activity of the individual units is summed to create a new macroscopic block variable, effectively halving the total number of variables in the system.
This step then repeated recursively $k$ times, clustering the $N/2^{k}$ previously formed blocks into progressively larger blocks at each subsequent scale. As this recursive process continues, different statistical observables can be measured across scales. Their behavior can be used to infer scale-invariant dynamics, mainly through power law statistics and shape collapse of some observables and the convergence of the probability distributions to a fixed non-Gaussian form. In Renormalization Group terminology, the RG transformation delineates a flow through the space of possible probability distributions, which, in the presence of scale invariance, trends toward a non-trivial fixed point in this space.

These multi-scale signatures can be used to analyze self-organized criticality, assess how dynamical states interact with anatomical or structural constraints, or serve as biological markers in empirical datasets.

#### References

1. Meshulam, L., Gauthier, J. L., Brody, C. D., Tank, D. W., & Bialek, W. (2019). Coarse Graining, Fixed Points, and Scaling in a Large Population of Neurons. *Physical Review Letters*, 123(17), 178103. [DOI: 10.1103/PhysRevLett.123.178103](https://doi.org/10.1103/PhysRevLett.123.178103)
2. Nicoletti, G., Suweis, S., & Maritan, A. (2020). Scaling and criticality in a phenomenological renormalization group. *Physical Review Research*, 2(2), 023144. [DOI: 10.1103/PhysRevResearch.2.023144](https://doi.org/10.1103/PhysRevResearch.2.023144)
3. Castro, D. M., Raposo, E. P., Copelli, M., & Santos, F. A. N. (2025). 
    Interdependent scaling exponents in the human brain. 
    *Physical Review Letters*, 135(19), 198401. 
    DOI: [10.1103/PhysRevLett.135.198401](https://doi.org/10.1103/lvwj-hjr3)
4. Cambrainha, G. G., Castro, D. M., Vasconcelos, N. A. P., Carelli, P. V., Copelli, M (2025).
    Criticality at Work: Scaling in the Mouse Cortex Enhances Performance
    *PRX Life 3*, 033026 
    DOI: [10.1103/w49n-2vz8](https://doi.org/10.1103/w49n-2vz8)

## Installation

To clone the repository and install the package locally, run:

```bash
git clone [https://github.com/Damiaom/prg_toolbox.git](https://github.com/Damiaom/prg_toolbox.git)
cd prg_toolbox
pip install .
```


### Dependencies

* NumPy
* Matplotlib
* SciPy (for minor alternative analysis)

## Quick Start

Here is a basic example showing how to initialize a coarse-graining analysis from a spike time array:

```python
import numpy as np
import matplotlib.pyplot as plt
import prg_toolbox as prg

# Set analysis parameters
prg_params = prg.config.AnalysisParams()
prg_params.rg_steps = 7
prg_params.observables = [prg.mean_variance, prg.activity_distribution]

# Load event timings (e.g. neuronal spiking data) with built-in methods
f = 'path_to_your_file_here'
timestamps = prg.tools.load_timestamps(f)
results = prg.tools.run_PRG(timestamps, user_params=prg_params)

# Load event timings with built-in methods
fig = plt.figure(figsize=(8,6))
prg.plot_mean_variance(results['mean_variance'])

# Load event timings with built-in methods
fig = plt.figure(figsize=(8,6))
prg.plot_mean_variance(results['activity_distribution'])


```
Detailed tutorials can be found in the Jupyter notebooks in the repository.
