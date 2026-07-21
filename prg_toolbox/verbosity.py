"""
Copyright (c) 2026 Daniel Miranda Castro. Licensed under the MIT License.

Shared verbosity helpers for the PRG analysis workflow.

`AnalysisParams.verbose` (and `CGVariables.verbose`) accept one of three levels:

    * "silent"   -- no prints, no warnings.
    * "warnings" -- only warning-style messages (e.g. dropped variables during
        coarse graining, silence clipping, misconfiguration notices). Default.
    * "full"     -- everything in "warnings", plus per-step timing and
        per-observable exponents printed as the analysis progresses.
"""
import time
import warnings
from contextlib import contextmanager

VERBOSITY_LEVELS = ("silent", "warnings", "full")


def validate_verbosity(verbose):
    if verbose not in VERBOSITY_LEVELS:
        raise ValueError(
            f"Invalid verbose option {verbose!r}. Choose from: {VERBOSITY_LEVELS}"
        )


def warn_if_verbose(message, verbose):
    if verbose != "silent":
        warnings.warn(message)


def print_if_full(message, verbose):
    if verbose == "full":
        print(message)


@contextmanager
def timed_step(label, verbose, indent=""):
    if verbose == "full":
        start = time.perf_counter()
        yield
        print(f"{indent}[timing] {label}: {time.perf_counter() - start:.3f}s")
    else:
        yield
