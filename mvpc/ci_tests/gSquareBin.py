# mvpc/ci_tests/gSquareBin.py

import numpy as np
from scipy.stats import chi2

def gSquareBin(x, y, S, data):
    """
    Unweighted G² test for binary variables.
    Faithful to R's gSquareBin (unweighted version).

    Parameters
    ----------
    x, y : int
        Indices of variables.
    S : list[int]
        Conditioning set.
    data : np.ndarray
        Binary data matrix (0/1), no missing values.

    Returns
    -------
    float
        p-value of the G² test.
    """
    # Extract relevant columns
    idx = [x, y] + list(S)
    sub = data[:, idx]

    # Ensure binary
    if not np.all((sub == 0) | (sub == 1)):
        raise ValueError("gSquareBin requires binary data (0/1).")

    n = sub.shape[0]
    d = len(S)

    # Number of parent configurations
    n_configs = 2 ** d

    # Build contingency table
    # Dimensions: 2 x 2 x (2^|S|)
    table = np.zeros((2, 2, n_configs))

    # Enumerate all S patterns
    if d > 0:
        S_patterns = np.array(list(np.ndindex(*(2,) * d)))
    else:
        S_patterns = np.zeros((1, 0), dtype=int)

    for i, pat in enumerate(S_patterns):
        if d > 0:
            mask = np.all(sub[:, 2:] == pat, axis=1)
        else:
            mask = np.ones(n, dtype=bool)

        subset = sub[mask]

        if subset.shape[0] == 0:
            continue

        # Count occurrences of x,y
        for xv in [0, 1]:
            for yv in [0, 1]:
                table[xv, yv, i] = np.sum((subset[:, 0] == xv) & (subset[:, 1] == yv))

    # Compute expected counts
    G2 = 0.0
    for k in range(n_configs):
        Nij = table[:, :, k]
        Nk = np.sum(Nij)

        if Nk == 0:
            continue

        row_sums = np.sum(Nij, axis=1)
        col_sums = np.sum(Nij, axis=0)

        expected = np.outer(row_sums, col_sums) / Nk

        # Avoid log(0)
        mask = (Nij > 0) & (expected > 0)
        G2 += 2 * np.sum(Nij[mask] * np.log(Nij[mask] / expected[mask]))

    # Degrees of freedom = 2^|S|
    df = 2 ** d

    return 1 - chi2.cdf(G2, df)
