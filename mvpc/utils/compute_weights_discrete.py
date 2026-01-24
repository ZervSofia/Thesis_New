"""
compute_weights_discrete.py

Faithful Python translation of R's compute.weights.discrete
used in binCItest.drw.

Implements:
    - indx_test_wise_deletion
    - get_ind_r_xys
    - compute_weights_discrete
"""

import numpy as np
from itertools import product

from .mvpc_utils import (
    test_wise_deletion,
    get_prt_i,
)


# ---------------------------------------------------------
# 1. Index version of test-wise deletion (R: indx_test_wise_deletion)
# ---------------------------------------------------------
def indx_test_wise_deletion(var_ind, data):
    mask = np.ones(data.shape[0], dtype=bool)
    for v in var_ind:
        mask &= ~np.isnan(data[:, v])
    return np.where(mask)[0]


# ---------------------------------------------------------
# 2. Identify missingness indicators among x,y,S,W (R: get_ind_r_xys)
# ---------------------------------------------------------
def get_ind_r_xys(ind, suffstat):
    data = suffstat["data"]
    ind_r = []
    for i in ind:
        if np.isnan(data[:, i]).any():
            ind_r.append(i)
    return ind_r


# ---------------------------------------------------------
# 3. Main function: compute discrete DRW weights
# ---------------------------------------------------------
def compute_weights_discrete(corr_ind, suffstat):
    """
    Faithful translation of R's compute.weights.discrete.

    corr_ind = indices of variables involved in the CI test
               (x, y, S, and W)
    """
    data = suffstat["data"]

    # Step 1: test-wise deletion indices
    idx_tw = indx_test_wise_deletion(corr_ind, data)
    n_tw = len(idx_tw)

    # Initialize weights = 1
    weights = np.ones(n_tw)

    # Step 2: identify missingness indicators among corr_ind
    ind_r = get_ind_r_xys(corr_ind, suffstat)

    # Step 3: for each missingness indicator, compute discrete density ratio
    for ind_ri in ind_r:
        prt_i = get_prt_i(ind_ri, suffstat)

        if len(prt_i) == 0:
            continue

        # Parent variable values
        pa = data[:, prt_i]

        # Values after deletion
        pa_del = pa[idx_tw]

        # Values with no missingness in parents
        mask_full = ~np.isnan(pa).any(axis=1)
        pa_full = pa[mask_full]

        # If insufficient data, skip
        if pa_full.shape[0] == 0 or pa_del.shape[0] == 0:
            continue

        # Enumerate all binary patterns of parents
        d = len(prt_i)
        patterns = np.array(list(product([0, 1], repeat=d)), dtype=int)

        # Count frequencies in full data
        counts_full = np.zeros(len(patterns))
        for i, pat in enumerate(patterns):
            counts_full[i] = np.sum(np.all(pa_full == pat, axis=1))

        # Count frequencies in deletion data
        counts_del = np.zeros(len(patterns))
        for i, pat in enumerate(patterns):
            counts_del[i] = np.sum(np.all(pa_del == pat, axis=1))

        # Convert to probabilities
        p_full = counts_full / np.sum(counts_full) if np.sum(counts_full) > 0 else np.ones(len(patterns)) / len(patterns)
        p_del = counts_del / np.sum(counts_del) if np.sum(counts_del) > 0 else np.ones(len(patterns)) / len(patterns)

        # Avoid division by zero
        p_del = np.where(p_del == 0, 1e-12, p_del)

        # Compute density ratio weights for each row in pa_del
        beta = np.zeros(n_tw)
        for i in range(n_tw):
            row = pa_del[i]
            idx = np.where(np.all(patterns == row, axis=1))[0][0]
            beta[i] = p_full[idx] / p_del[idx]

        # Multiply into global weights
        weights *= beta

    return weights
