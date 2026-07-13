# The coarse graining method (`coarse_graining.py`)

The coarse-graining method within the PRG framework operates by iteratively grouping a high-dimensional system's variables to observe how statistical properties change across scales. Instead of grouping units based on their physical location on a rigid grid (as with the traditional  real-space RG from physics theory), the method groups units based on their functional similarity. The distance between two variables is defined by their statistical correlation, meaning that highly correlated units are treated as neighbors in an abstract functional space and clustered together into a block.

The method begins with $N$ individual binary variables, which the algorithm clusters into pairs or blocks. For each block, the underlying activity of the individual units is summed to create a new macroscopic block variable, effectively halving the total number of variables in the system. The newly $N/2$ formed coarse-grained variables $\{x^{(k+1)}\}$ for the subsequent scale are then defined as

$x_i^{(k+1)} = x_i^{(k)} + x_{j*i}^{(k)} \quad ,$

where $x_i^{(k)}$ and $x_{j*i}^{(k)}$ represent the time series activity of two maximally correlated variables at renormalization step $k$.

This step then repeated recursively, clustering the $N/2^{k}$ previously formed blocks into progressively larger blocks at each subsequent scale. As this recursive process continues, different statistical observables can be measured across scales. Their behavior can be used to infer scale-invariant dynamics, mainly through power law statistics and shape collapse of some observables and the convergence of the probability distributions to a fixed non-Gaussian form. In Renormalization Group terminology, the RG transformation delineates a flow through the space of possible probability distributions, which, in the presence of scale invariance, trends toward a non-trivial fixed point in this space.

-------------------------------

::: prg_toolbox.coarse_graining
