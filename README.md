
# Phenomenological Renormalization Group (PRG) Toolbox

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This toolbox applies the coarse graining procedure (also called Phenomenological Renormalization Group) introduced by [Meshulam et al. (2019)](https://doi.org/10.1103/PhysRevLett.123.178103) to multidimensional binary time series. It computes and stores the coarse-grained variables and subsequent observables.
The PRG procedure operates in a model-independent manner directly on empirical datasets. By systematically coarse-graining population activity, the it maps the flow of collective statistical properties across successive spatio-temporal scales to probe for signatures of scale-invariant dynamics in data.

## Installation

You can clone and install the toolbox locally from GitHub:

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
prg_params.observables = [prg.mean_variance, prg.covariance_spectrum]

f = 'path_to_your_file_here'
timestamps = prg.tools.load_timestamps(f)
results = prg.tools.run_PRG(timestamps, user_params=prg_params)

fig = plt.figure(figsize=(8,6))
prg.plot_mean_variance(results['mean_variance'])

fig = plt.figure(figsize=(8,6))
prg.plot_mean_variance(results['covariance_spectrum'])

```
Detailed tutorials can be found in the Jupyter notebooks in the repository.
