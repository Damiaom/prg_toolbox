# Configuration (`config.py`)

This module provides a centralized parameter management system for the analysis pipeline. It utilizes typed Python dataclasses to define and store the operational settings required across the entire toolbox, so that structural processing options and visualization preferences are maintained without requiring modifications to the underlying code.

The module groups related parameters into specific structural categories. For temporal preprocessing, it defines the duration of steady-state observation windows and the precise temporal resolution used to discretize continuous event streams into binary matrices. For spatial evaluation, it manages subsampling parameters, establishing the size of the unit subsets and the number of random replication trials used to calculate average scaling metrics.

Additionally, the configuration layer can also receive matplotlib kwargs, which provides some control over the result visualization without modifying the main code. These individual parameter categories are all nested within a primary analysis configuration object, which is passed systematically through the temporal slicing tools, coarse-graining loops, and observable tracking functions.

-----------------------------

::: prg_toolbox.config