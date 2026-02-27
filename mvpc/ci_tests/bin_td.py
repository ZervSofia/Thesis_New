import numpy as np
from scipy.stats import ttest_ind
from ..utils.mvpc_utils import test_wise_deletion

def bin_ci_td(x, y, S, suffstat):
    """
    Deletion-based binary-continuous CI test.
    faithful to R's binCItest.td (ANOVA-style test).
    """
    data = suffstat["data"]
    idx = [x, y] + list(S)

    # test-wise deletion
    sub = test_wise_deletion(idx, data)
    if sub.shape[0] < 5:
        return 1.0

    # Extract variables
    b = sub[:, 0]          
    c = sub[:, 1]          
    S_mat = sub[:, 2:]     

    # if no conditioning set, simple t-test
    if S_mat.shape[1] == 0:
        g0 = c[b == 0]
        g1 = c[b == 1]
        if len(g0) < 2 or len(g1) < 2:
            return 1.0
        _, p = ttest_ind(g0, g1, equal_var=False)
        return p if p is not None else 1.0

    # regress c on S (no intercept)
    try:
        beta, *_ = np.linalg.lstsq(S_mat, c, rcond=None)
    except np.linalg.LinAlgError:
        return 1.0

    r = c - S_mat @ beta  # residuals

    # t-test on residuals
    g0 = r[b == 0]
    g1 = r[b == 1]
    if len(g0) < 2 or len(g1) < 2:
        return 1.0

    _, p = ttest_ind(g0, g1, equal_var=False)
    return p if p is not None else 1.0
