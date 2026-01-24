"""
gauss_drw.py

Faithful Python translation of R's gaussCItest.drw.
Implements the DRW (density ratio weighting) correction for Gaussian CI tests.
"""

import numpy as np
from numpy.linalg import inv
from scipy.stats import norm

# ---------------------------------------------------------
# Import your MVPC utilities (adjust path if needed)
# ---------------------------------------------------------
from ..utils.mvpc_utils import (
    test_wise_deletion,
    cond_PermC,
    get_prt_m_xys,
)
from ..utils.compute_weights_continuous import compute_weights_continuous

# (We will create weights_continuous.py next if you don’t have it yet)


# ---------------------------------------------------------
# 1. Weighted covariance (faithful to R's wtd.cors)
# ---------------------------------------------------------
def weighted_cov(X, weights):
    """
    Compute weighted covariance matrix.
    Equivalent to R's wtd.cors(X, X, weights).
    """
    w = np.asarray(weights)
    w = w / np.sum(w)

    mean_w = np.sum(X * w[:, None], axis=0)
    X_centered = X - mean_w

    cov_w = (X_centered * w[:, None]).T @ X_centered
    return cov_w


# ---------------------------------------------------------
# 2. Weighted Gaussian CI test (R: gaussCItest with weighted C)
# ---------------------------------------------------------
def gauss_ci_weighted(x, y, S, C, n_eff):
    """
    Gaussian CI test using a weighted covariance matrix C.
    Mirrors R's gaussCItest but uses weighted covariance.
    """
    # Extract submatrix for variables [x, y, S]
    idx = [x, y] + list(S)
    C_sub = C[np.ix_(idx, idx)]

    try:
        prec = inv(C_sub)
    except np.linalg.LinAlgError:
        return 1.0

    # Partial correlation
    r_xy_S = -prec[0, 1] / np.sqrt(prec[0, 0] * prec[1, 1])
    r_xy_S = np.clip(r_xy_S, -0.999999, 0.999999)

    # Fisher z-transform
    z = 0.5 * np.log((1 + r_xy_S) / (1 - r_xy_S))
    stat = np.sqrt(n_eff - len(S) - 3) * abs(z)

    return 2 * (1 - norm.cdf(stat))


# ---------------------------------------------------------
# 3. DRW-corrected Gaussian CI test (R: gaussCItest.drw)
# ---------------------------------------------------------
def gauss_ci_drw(x, y, S, suffstat):
    """
    Faithful Python translation of R's gaussCItest.drw.

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
        # Fall back to deletion-based Gaussian CI
        from .gauss_permc import gauss_ci_td
        return gauss_ci_td(x, y, S, suffstat)

    # -----------------------------------------------------
    # Step 2: Get parents of missingness indicators of {x,y,S}
    # -----------------------------------------------------
    ind_test = [x, y] + list(S)
    ind_W = list(set(get_prt_m_xys(ind_test, suffstat)))

    if len(ind_W) == 0:
        from .gauss_permc import gauss_ci_td
        return gauss_ci_td(x, y, S, suffstat)

    # -----------------------------------------------------
    # Step 3: Recursively add parents of W until closure
    # -----------------------------------------------------
    pa_W = list(set(get_prt_m_xys(ind_W, suffstat)))
    candi_W = list(set(pa_W) - set(ind_W))

    while len(candi_W) > 0:
        ind_W = list(set(ind_W) | set(candi_W))
        pa_W = list(set(get_prt_m_xys(ind_W, suffstat)))
        candi_W = list(set(pa_W) - set(ind_W))

    ind_W = list(set(ind_W))

    # -----------------------------------------------------
    # Step 4: Compute DRW weights (faithful to R)
    # -----------------------------------------------------
    corr_ind = ind_test + ind_W
    weights = compute_weights_continuous(corr_ind, suffstat)

    # -----------------------------------------------------
    # Step 5: Test-wise deletion on corr_ind
    # -----------------------------------------------------
    data_tw = test_wise_deletion(corr_ind, data)
    weights_tw = weights


    # -----------------------------------------------------
    # Step 6: Weighted covariance
    # -----------------------------------------------------
    C_w = weighted_cov(data_tw[:, corr_ind], weights_tw)
    n_eff = np.sum(weights_tw)

    # -----------------------------------------------------
    # Step 7: Gaussian CI test using weighted covariance
    # -----------------------------------------------------
    # In the weighted covariance matrix, x,y,S correspond to indices 0,1,2,...
    S_local = list(range(2, 2 + len(S)))
    return gauss_ci_weighted(0, 1, S_local, C_w, n_eff)
