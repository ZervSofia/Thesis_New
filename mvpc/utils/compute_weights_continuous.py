"""
compute_weights_continuous.py

Faithful Python translation of R's compute.weights.continuous
used in gaussCItest.drw.

Implements:
    - indx_test_wise_deletion
    - get_ind_r_xys
    - get_prt_i
    - get_logidata
    - kde_weights
    - compute_weights_continuous
"""

import numpy as np
from scipy.stats import gaussian_kde

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
# 2. Identify missingness indicators among x,y,S (R: get_ind_r_xys)
# ---------------------------------------------------------
def get_ind_r_xys(ind, suffstat):
    data = suffstat["data"]
    ind_r = []
    for i in ind:
        if np.isnan(data[:, i]).any():
            ind_r.append(i)
    return ind_r


# ---------------------------------------------------------
# 3. Logistic regression data builder (R: get_logidata)
# ---------------------------------------------------------
def get_logidata(ind_ri, suffstat):
    """
    Build the logistic regression dataset for missingness indicator ri.
    R version:
        ri <- as.integer(!is.na(data[, ind_ri]))
        logidata <- data.frame(ri, data[, prt_i])
        test_wise_deletion(...)
    """
    data = suffstat["data"]
    prt_i = get_prt_i(ind_ri, suffstat)

    ri = (~np.isnan(data[:, ind_ri])).astype(int)

    if len(prt_i) == 0:
        X = ri.reshape(-1, 1)
    else:
        X = np.column_stack([ri, data[:, prt_i]])

    # test-wise deletion on all columns
    idx = indx_test_wise_deletion(range(X.shape[1]), X)
    return X[idx, :]


# ---------------------------------------------------------
# 4. KDE-based density ratio weights (R: kde.weights)
# ---------------------------------------------------------
def kde_weights(x_del, x_full):
    """
    R version:
        f_w  = density(x_full)
        f_wr = density(x_del)
        beta = f_w(x_del) / f_wr(x_del)
    """
    if len(x_full) < 5 or len(x_del) < 5:
        return np.ones_like(x_del)

    kde_full = gaussian_kde(x_full)
    kde_del = gaussian_kde(x_del)

    f_full = kde_full.evaluate(x_del)
    f_del = kde_del.evaluate(x_del)

    # Avoid division by zero
    f_del = np.where(f_del == 0, 1e-12, f_del)

    return f_full / f_del


# ---------------------------------------------------------
# 5. Main function: compute continuous DRW weights
# ---------------------------------------------------------
def compute_weights_continuous(corr_ind, suffstat):
    """
    Faithful translation of R's compute.weights.continuous.

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

    # Step 3: for each missingness indicator, compute density ratio weights
    for ind_ri in ind_r:
        prt_i = get_prt_i(ind_ri, suffstat)

        # Parent variable values
        pa = data[:, prt_i] if len(prt_i) > 0 else None

        # Values after deletion
        if pa is None:
            continue

        pa_del = pa[idx_tw]
        pa_full = pa[~np.isnan(pa).any(axis=1)]

        # Compute KDE-based density ratio
        beta = kde_weights(pa_del.flatten(), pa_full.flatten())

        # Multiply into global weights
        weights *= beta

    return weights
