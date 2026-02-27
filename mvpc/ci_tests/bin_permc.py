"""
bin_permc.py

translation of binCItest.permc.
PermC correction for binary CI tests.
"""

import numpy as np
from itertools import product

from ..utils.mvpc_utils import (
    test_wise_deletion,
    cond_PermC,
    get_prt_m_xys,
    perm,
)

from .bin_td import bin_ci_td      
from .gSquareBin import gSquareBin # unweighted G^2 test



# generate all binary combinations
def binary_combinations(k):
    return np.array(list(product([0, 1], repeat=k)), dtype=int)



# estimate multivariate Bernoulli distr
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



def sample_mult_binary(n, p, patterns):
    """
    Sample n rows from a multivariate Bernoulli distribution.
    """
    idx = np.random.choice(len(p), size=n, p=p)
    return patterns[idx]



# PermC for binary CI test

def bin_ci_permc(x, y, S, suffstat):
    """
    binCItest.permc.

    suffstat must contain:
        - "data": full dataset
        - "prt_m": missingness-parent structure
        - "skel": initial skeleton (for cond_PermC)
    """
    data = suffstat["data"]

    # Check if correction is needed
    if not cond_PermC(x, y, S, suffstat):
        return bin_ci_td(x, y, S, suffstat)

    
    # Identify W = parents of missingness indicators
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

    
    # Build index set for PermC
    ind_permc = ind_test + ind_W


    # Test-wise deletion
    data_tw = test_wise_deletion(ind_permc, data)
    data_tw = data_tw[:, ind_permc]

    n_tw = data_tw.shape[0]
    d_test = len(ind_test)
    d_W = len(ind_W)

 
    # Estimate CPDs for each W pattern
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


    # Shuffle W 
    data_W_perm = perm(ind_W, data)
    data_W_perm = np.asarray(data_W_perm)

    # Match number of rows to n_tw
    if data_W_perm.shape[0] < n_tw:
        W_perm = data_W_perm[:n_tw, :]
    else:
        W_perm = data_W_perm[:n_tw, :]


    # Generate virtual data
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


    # standard binary CI test on virtual data
    if len(ind_test) > 2:
        S_local = list(range(2, len(ind_test)))
    else:
        S_local = []

    return gSquareBin(0, 1, S_local, data_vir)
