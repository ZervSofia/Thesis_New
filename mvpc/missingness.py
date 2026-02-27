

"""
We Implement:
    - get_m_ind: detect which variables contain missing values
    - get_prt_R_ind: identify parents of each missingness indicator
    - detection_prt_m: orchestrate missingness-parent detection

This module corresponds to the first step of MVPC.
"""

import numpy as np
from itertools import combinations
from tqdm import tqdm



def get_m_ind(data):
    """
    Identify variables that contain missing values.

    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (n_samples, n_variables).

    Returns
    -------
    list[int]
        Indices of variables with at least one missing value.
    """
    return [j for j in range(data.shape[1]) if np.isnan(data[:, j]).any()]


def get_prt_R_ind(data, indep_test, alpha, R_ind):
    """
    PC-style parent detection for the missingness indicator of R_ind.

    This mirrors the logic of the R function get_prt_R_ind:
        1. Replace column R_ind with a binary missingness indicator.
        2. Initialize a fully connected undirected graph over all variables.
        3. Run a PC-style skeleton search, but:
            - only test edges involving R_ind,
            - use neighbors of R_ind as conditioning candidates.
        4. Variables that remain adjacent to R_ind are its parents.

    Parameters
    ----------
    data : np.ndarray
        Original data matrix (n x p).
    indep_test : callable
        Function (x, y, S, data) -> p-value.
    alpha : float
        Significance threshold.
    R_ind : int
        Index of the variable whose missingness indicator is analyzed.

    Returns
    -------
    list[int]
        Parent variables of the missingness indicator.
    """
    n, p = data.shape

    # Create missingness indicator and replace column
    data_mod = data.copy()
    data_mod[:, R_ind] = np.isnan(data[:, R_ind]).astype(int)

    # Initialize fully connected undirected graph (no self-loops)
    G = np.ones((p, p), dtype=bool)
    np.fill_diagonal(G, False)

    # PC-style loop over conditioning set size
    ord_size = 0
    done = False

    while not done and G[R_ind].any():
        done = True

        # Current neighbors of R_ind
        neighbors = np.where(G[R_ind])[0]

        if len(neighbors) < ord_size:
            break

        if len(neighbors) > ord_size:
            done = False

        for y in tqdm(list(neighbors), desc=f"R={R_ind}, ord={ord_size}", leave=False):
            # neighbors of R_ind excluding y
            nbrs = [k for k in neighbors if k != y]

            if len(nbrs) < ord_size:
                continue

            independent = False

            for S in combinations(nbrs, ord_size):
                suffstat = {"data": data_mod}
                pval = indep_test(R_ind, y, S, suffstat)


                if pval >= alpha:
                    # remove edge if independent
                    G[R_ind, y] = G[y, R_ind] = False
                    independent = True
                    break

            if independent:
                continue

        ord_size += 1

    # Parents = variables still adjacent to R_ind
    parents = list(np.where(G[R_ind])[0])
    return parents


def detection_prt_m(data, indep_test, alpha, p):
    
    m_inds = get_m_ind(data)
    print(f"[Step 1] m_inds (vars with NaNs): {m_inds}")
    prt = {}

    for R_ind in tqdm(m_inds, desc="Detecting parents of missingness indicators"):
        parents = get_prt_R_ind(data, indep_test, alpha, R_ind)
        print(f"[Step 1] R_ind={R_ind}, parents={parents}")
        if parents:
            prt[R_ind] = parents

    m_inds_filtered = [m for m in m_inds if m in prt]
    print(f"[Step 1] m_inds_filtered (with ≥1 parent): {m_inds_filtered}")
    print(f"[Step 1] prt dict: {prt}")

    return {"m": m_inds_filtered, "prt": prt}
