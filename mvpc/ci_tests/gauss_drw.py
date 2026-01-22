"""
gauss_drw.py

Density Ratio Weighted Gaussian CI test for MVPC.

Idea:
    - Missingness depends on some parent variables (from prt_m).
    - We estimate weights that correct for this selection bias.
    - Then we run a *weighted* Gaussian CI test using those weights.
"""

import numpy as np
from scipy.stats import norm
from numpy.linalg import inv
from sklearn.linear_model import LogisticRegression


def _gaussian_ci_stat_weighted(x, y, S, data, weights):
    """
    Weighted Fisher-z statistic for Gaussian CI test.
    """
    vars_idx = [x, y] + list(S)
    sub = data[:, vars_idx]
    w = weights.copy()

    # Keep only rows without missing values
    mask = ~np.isnan(sub).any(axis=1)
    sub = sub[mask]
    w = w[mask]

    if sub.shape[0] < 5:
        return 0.0

    # Normalize weights
    w = w / w.sum()

    # Weighted covariance
    mean = np.average(sub, axis=0, weights=w)
    xc = sub - mean
    cov = (w[:, None] * xc).T @ xc

    try:
        prec = inv(cov)
    except np.linalg.LinAlgError:
        return 0.0

    r_xy_S = -prec[0, 1] / np.sqrt(prec[0, 0] * prec[1, 1])
    z = 0.5 * np.log((1 + r_xy_S) / (1 - r_xy_S))
    return abs(z)


def _gaussian_ci_stat_unweighted(x, y, S, data):
    """
    Standard (unweighted) Fisher-z statistic.
    """
    vars_idx = [x, y] + list(S)
    sub = data[:, vars_idx]
    sub = sub[~np.isnan(sub).any(axis=1)]

    if sub.shape[0] < 5:
        return 0.0

    cov = np.cov(sub, rowvar=False)
    try:
        prec = inv(cov)
    except np.linalg.LinAlgError:
        return 0.0

    r_xy_S = -prec[0, 1] / np.sqrt(prec[0, 0] * prec[1, 1])
    z = 0.5 * np.log((1 + r_xy_S) / (1 - r_xy_S))
    return abs(z)


def _compute_weights_for_variable(data, target, prt_m):
    """
    Compute DRW weights for a variable with missingness.

    We model P(R = 1 | parents) via logistic regression,
    then derive inverse-probability-like weights for observed entries.
    """
    n, p = data.shape
    m_indicator = np.isnan(data[:, target]).astype(int)

    parents = prt_m["prt"].get(target, [])
    if len(parents) == 0:
        return np.ones(n)

    Z = data[:, parents]
    mask = ~np.isnan(Z).any(axis=1)
    Z_obs = Z[mask]
    R_obs = m_indicator[mask]

    if Z_obs.shape[0] < 10 or len(np.unique(R_obs)) < 2:
        return np.ones(n)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Z_obs, R_obs)
    prob = clf.predict_proba(Z_obs)[:, 1]

    eps = 1e-6
    prob = np.clip(prob, eps, 1 - eps)

    w_obs = 1.0 / (1.0 - prob)
    weights = np.ones(n)
    weights[mask] = w_obs

    return weights


def gauss_ci_drw(x, y, S, suffstat):
    """
    DRW-corrected Gaussian CI test.

    suffstat must contain:
        - "data": np.ndarray (n x p)
        - "prt_m": missingness-parent structure (optional but needed for DRW)
    """
    data = suffstat["data"]
    prt_m = suffstat.get("prt_m", None)

    # If no missingness-parent info, fall back to unweighted test
    if prt_m is None:
        z = _gaussian_ci_stat_unweighted(x, y, S, data)
        return 2 * (1 - norm.cdf(z))

    missing_inds = prt_m["m"]

    # If neither variable has missingness parents, use standard Gaussian CI
    if x not in missing_inds and y not in missing_inds:
        z = _gaussian_ci_stat_unweighted(x, y, S, data)
        return 2 * (1 - norm.cdf(z))

    n = data.shape[0]
    weights = np.ones(n)

    if x in missing_inds:
        w_x = _compute_weights_for_variable(data, x, prt_m)
        weights *= w_x

    if y in missing_inds:
        w_y = _compute_weights_for_variable(data, y, prt_m)
        weights *= w_y

    z = _gaussian_ci_stat_weighted(x, y, S, data, weights)
    p = 2 * (1 - norm.cdf(z))
    return p
