# mvpc/ci_tests/bin_td.py

import numpy as np
from ..utils.mvpc_utils import test_wise_deletion
from .gSquareBin import gSquareBin

def bin_ci_td(x, y, S, suffstat):
    """
    Deletion-based binary CI test.
    Faithful to R's binCItest.td.

    Parameters
    ----------
    x, y : int
        Variable indices.
    S : list[int]
        Conditioning set.
    suffstat : dict
        Must contain "data".

    Returns
    -------
    float
        p-value of the G² test.
    """
    data = suffstat["data"]
    idx = [x, y] + list(S)

    # Test-wise deletion
    sub = test_wise_deletion(idx, data)

    # Extract only the relevant columns
    sub = sub[:, idx]

    # If too few samples, return independence
    if sub.shape[0] < 5:
        return 1.0

    # Run unweighted G² test
    return gSquareBin(0, 1, list(range(2, len(idx))), sub)
