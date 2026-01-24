"""
bin_permc.py

Faithful Python translation of R's binCItest.permc.
Implements the PermC correction for binary conditional independence tests.
"""

import numpy as np
from itertools import product

from ..utils.mvpc_utils import (
    test_wise_deletion,
    cond_PermC,
    get_prt_m_xys,
    perm,
)

from .bin_td import bin_ci_td      # your deletion-based binary CI test
from .gSquareBin import gSquareBin # your unweighted G² test


# ---------------------------------------------------------
# Helper: generate all binary combinations of length k
# (R: bincombinations)
# ---------------------------------------------------------
def binary_combinations(k):
    return np.array(list(product([0, 1], repeat=k)), dtype=int)


# ---------------------------------------------------------
# Helper: estimate multivariate Bernoulli distribution
# (R: ObtainMultBinaryDist)
# ---------------------------------------------------------
def estimate_mult_binary_dist(data):
    """
    Estimate joint distribution of multivariate binary vector.
    Returns:
        - p: probability of each binary pattern (2^d vector)
        - patterns: matrix of all patterns (2^d x d)
    """
    n, d = data.shape
    patterns = binary_combinations(d)
    counts = np.zeros(len(patterns))

    for i, pat in enumerate(patterns):
        mask = np.all(data == pat, axis=1)
        counts[i] = np.sum(mask)

    p = counts / n if n > 0 else np.ones(len(patterns)) / len(patterns)
    return p, patterns


# ---------------------------------------------------------
# Helper: sample from multivariate Bernoulli distribution
# (R: RMultBinary)
# ---------------------------------------------------------
def sample_mult_binary(n, p, patterns):
    """
    Sample n rows from a multivariate Bernoulli distribution.
    """
    idx = np.random.choice(len(p), size=n, p=p)
    return patterns[idx]


# ---------------------------------------------------------
# Main function: PermC for binary CI test
# ---------------------------------------------------------
def bin_ci_permc(x, y, S, suffstat):
    """
    Faithful Python translation of R's binCItest.permc.

    suffstat must contain:
        - "data": full dataset
        - "prt_m": missingness-parent structure
        - "skel": initial skeleton (for cond_PermC)
    """
    data = suffstat["data"]

    # -----------------------------------------------------
    # Step 1: Check if correction is needed
    # -----------------------------------------------------
    if not cond_PermC(x, y, S, suffstat):
        return bin_ci_td(x, y, S, suffstat)

    # -----------------------------------------------------
    # Step 2: Identify W = parents of missingness indicators
    # -----------------------------------------------------
    ind_test = [x, y] + list(S)
    ind_W = list(set(get_prt_m_xys(ind_test, suffstat)))

    if len(ind_W) == 0:
        return bin_ci_td(x, y, S, suffstat)

    # Recursively add parents of W
    pa_W = list(set(get_prt_m_xys(ind_W, suffstat)))
    candi_W = list(set(pa_W) - set(ind_W))

    while len(candi_W) > 0:
        ind_W = list(set(ind_W) | set(candi_W))
        pa_W = list(set(get_prt_m_xys(ind_W, suffstat)))
        candi_W = list(set(pa_W) - set(ind_W))

    ind_W = list(set(ind_W))

    # -----------------------------------------------------
    # Step 3: Build index set for PermC
    # -----------------------------------------------------
    ind_permc = ind_test + ind_W

    # -----------------------------------------------------
    # Step 4: Test-wise deletion
    # -----------------------------------------------------
    data_tw = test_wise_deletion(ind_permc, data)
    data_tw = data_tw[:, ind_permc]

    n_tw = data_tw.shape[0]
    d_test = len(ind_test)
    d_W = len(ind_W)

    # -----------------------------------------------------
    # Step 5: Estimate CPDs for each W pattern
    # -----------------------------------------------------
    W_patterns = binary_combinations(d_W)
    n_patterns = len(W_patterns)

    joint_dists = []

    for pat in W_patterns:
        mask = np.all(data_tw[:, d_test:] == pat, axis=1)
        subset = data_tw[mask][:, :d_test]

        if subset.shape[0] == 0:
            # uniform distribution if no data
            p = np.ones(2**d_test) / (2**d_test)
            patterns = binary_combinations(d_test)
        else:
            p, patterns = estimate_mult_binary_dist(subset)

        joint_dists.append((p, patterns))

    # -----------------------------------------------------
    # Step 6: Shuffle W (R: perm)
    # -----------------------------------------------------
    data_W_perm = perm(ind_W, data)
    data_W_perm = np.asarray(data_W_perm)

    # Match number of rows to n_tw
    if data_W_perm.shape[0] < n_tw:
        W_perm = data_W_perm[:n_tw, :]
    else:
        W_perm = data_W_perm[:n_tw, :]

    # -----------------------------------------------------
    # Step 7: Generate virtual data
    # -----------------------------------------------------
    virtual_rows = []

    for i, pat in enumerate(W_patterns):
        mask = np.all(W_perm == pat, axis=1)
        count = np.sum(mask)

        if count == 0:
            continue

        p, patterns = joint_dists[i]
        samples = sample_mult_binary(count, p, patterns)
        virtual_rows.append(samples)

    if len(virtual_rows) == 0:
        return bin_ci_td(x, y, S, suffstat)

    data_vir = np.vstack(virtual_rows)

    # -----------------------------------------------------
    # Step 8: Run standard binary CI test on virtual data
    # -----------------------------------------------------
    # In virtual data: col 0 = x, col 1 = y, col 2.. = S
    if len(ind_test) > 2:
        S_local = list(range(2, len(ind_test)))
    else:
        S_local = []

    return gSquareBin(0, 1, S_local, data_vir)
