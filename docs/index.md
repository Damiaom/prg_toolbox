# PRG Toolbox

PRG Toolbox is a Python package that implements the Phenomenological Renormalization Group (PRG) coarse graining method introduced by Meshulam et al. (2019) [1,2]. It performs model-free analysis designed to study scale-invariant behavior and collective dynamics in high-dimensional multivariate time series, such as multi-neuron spike recordings or rs-fMRI data [3,4].

## Method Overview

The toolbox applies the intuition of the renormalization group from statistical physics directly to empirical data without relying on an explicit underlying model. Near a second-order phase transition, the collective dynamics of a system become insensitive to most microscopic details, displaying scale-invariant activity. 

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

## Documentation Index

The toolbox is structured around configurations, core renormalization routines, individual metric implementations, and pipeline orchestration.

### 📑 Configuration (`config.py`)
Contains the dataclasses used to set parameters for the analysis. This includes time windowing settings (`TimeWindowingParams`), spatial subsampling choices (`SubsamplingParams`), and matplotlib figure layout arguments (`PlotStyleConfig`).

### 🔍 Coarse-Graining (`coarse_graining.py`)
Implements the `CGVariables` class, which handles the real-space Renormalization Group transformation. It executes the clustering loops, applies similarity matching to pair variables step-by-step, and maps the constituent unit lineages across scales.

### 📈 Statistical Metrics (`observables.py`)
Implements the distinct diagnostic classes used to calculate network scaling variables at each coarse-graining step. Each class (such as variance scaling, silence probability, eigenvalue spectrum, or temporal decay) accepts a `CGVariables` object and evaluates specific scaling behaviors and exponents.

### 🛠️ Analysis Tools (`analysis_tools.py`)
Provides functions for data ingestion, temporal preprocessing, and baseline calculations. It includes utilities to remove initial data transients, slice spike timestamps into temporal windows, binarize event arrays, generate shuffled inter-spike interval (ISI) data, and average results across trials.

### ⚙️ Pipelines (`pipelines.py`)
Coordinates workflow execution by combining the parameter settings, data tools, and coarse-graining routines. It contains functions to process a single data file (`run_PRG`) or sweep folders sequentially or in parallel using multi-core processing.

### 🎨 Visualizations (`plotting/`)
A submodule directory containing plotting functions corresponding to the classes in `observables.py`. These functions import style settings from the configuration classes to plot uniform output figures.

## Installation

To clone the repository and install the package locally, run:

```bash
git clone [https://github.com/Damiaom/prg_toolbox.git](https://github.com/Damiaom/prg_toolbox.git)
cd prg_toolbox
pip install .
```

## Quick Start

Here is a basic example showing how to initialize a coarse-graining analysis from a spike time array:

```python
import matplotlib.pyplot as plt
import prg_toolbox as prg

# Set analysis parameters
prg_params = prg.config.AnalysisParams()
prg_params.rg_steps = 7
prg_params.observables = [prg.mean_variance, prg.covariance_spectrum]

# Load event timings (e.g. neuronal spiking data) with built-in methods
f = 'path_to_your_file_here'
timestamps = prg.tools.load_timestamps(f)
results = prg.tools.run_PRG(timestamps, user_params=prg_params)

# Plot results
fig = plt.figure(figsize=(8,6))
prg.plot_mean_variance(results['mean_variance'])

fig = plt.figure(figsize=(8,6))
prg.plot_mean_variance(results['covariance_spectrum'])

```
Detailed tutorials can be found in the Jupyter notebooks in the repository.