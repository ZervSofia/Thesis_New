"""
bin_drw.py

translation of binCItest.drw.
DRW correction for binary CI tests.
"""

import numpy as np

from ..utils.mvpc_utils import (
    test_wise_deletion,
    cond_PermC,
    get_prt_m_xys,
)

from .bin_td import bin_ci_td
from .gSquareBin import gSquareBin
from ..utils.compute_weights_discrete import compute_weights_discrete


# Weighted G^2 test (binary)
def gSquareBin_weighted(x, y, S, data, weights):
    """
    Weighted version of gSquareBin.
    from gSquareBin.weighted.
    """

    idx = [x, y] + list(S)
    sub = data[:, idx]
    w = weights

    # Ensure binary
    if not np.all((sub == 0) | (sub == 1)):
        raise ValueError("gSquareBin_weighted requires binary data")

    n = sub.shape[0]
    d = len(S)
    n_configs = 2 ** d

    # Build contingency table
    table = np.zeros((2, 2, n_configs))

    if d > 0:
        S_patterns = np.array(list(np.ndindex(*(2,) * d)))
    else:
        S_patterns = np.zeros((1, 0), dtype=int)

    for k, pat in enumerate(S_patterns):
        if d > 0:
            mask = np.all(sub[:, 2:] == pat, axis=1)
        else:
            mask = np.ones(n, dtype=bool)

        subset = sub[mask]
        w_sub = w[mask]

        if subset.shape[0] == 0:
            continue

        for xv in [0, 1]:
            for yv in [0, 1]:
                table[xv, yv, k] = np.sum(w_sub[(subset[:, 0] == xv) & (subset[:, 1] == yv)])

    # Compute G^2
    G2 = 0.0
    for k in range(n_configs):
        Nij = table[:, :, k]
        Nk = np.sum(Nij)

        if Nk == 0:
            continue

        row_sums = np.sum(Nij, axis=1)
        col_sums = np.sum(Nij, axis=0)

        expected = np.outer(row_sums, col_sums) / Nk

        mask = (Nij > 0) & (expected > 0)
        G2 += 2 * np.sum(Nij[mask] * np.log(Nij[mask] / expected[mask]))

    df = 2 ** d
    from scipy.stats import chi2
    return 1 - chi2.cdf(G2, df)


# DRW-corrected binary CI test
def bin_ci_drw(x, y, S, suffstat):
    """
    binCItest.drw.

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


    # Compute DRW weights (discrete)
    corr_ind = ind_test + ind_W
    weights = compute_weights_discrete(corr_ind, suffstat)


    # Test-wise deletion
    data_tw = test_wise_deletion(corr_ind, data)
    weights_tw = weights


    # Weighted G^2 test
    S_local = list(range(2, 2 + len(S)))
    return gSquareBin_weighted(0, 1, S_local, data_tw[:, corr_ind], weights_tw)
