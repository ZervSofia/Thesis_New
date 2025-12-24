import numpy as np
from scipy.stats import norm

# ============================================================
# 1. Choose missingness indicators (MAR)
#    Corresponds to create_mar_ind() in R
# ============================================================

def create_mar_ind(colliders,
                   collider_parents,
                   num_var: int,
                   num_extra_e: int = 3,
                   num_m: int = 6,
                   seed: int | None = None):
    """
    Select variables with missing values (ms) and
    their missingness parents (prt_ms), following the R logic.
    """
    if seed is not None:
        np.random.seed(seed)

    ms = []
    prt_ms = []

    # Step 1: parents of colliders
    for cl, parents in zip(colliders, collider_parents):
        for p in parents:
            if (p not in ms) and (p not in prt_ms):
                ms.append(p)
                prt_ms.append(cl)
            if len(ms) >= num_extra_e:
                break
        if len(ms) >= num_extra_e:
            break

    # Step 2: randomly fill remaining missing variables
    remaining_prt = list(set(range(num_var)) - set(prt_ms))
    remaining_prt = [v for v in remaining_prt if v not in ms]
    np.random.shuffle(remaining_prt)

    remaining_ms = list(set(range(num_var)) - set(ms) - set(prt_ms))
    np.random.shuffle(remaining_ms)

    n_remaining = num_m - len(ms)

    for i in range(n_remaining):
        prt_ms.append(remaining_prt[i])
        ms.append(remaining_ms[i])

    return ms, prt_ms


# ============================================================
# 2. Generate MAR missing values
#    Corresponds to generate_missing_values() + mis_cal_ind()
# ============================================================

def generate_missing_values(X_complete: np.ndarray,
                            ms: list[int],
                            prt_ms: list[int],
                            p_missing_h: float = 0.9,
                            p_missing_l: float = 0.01,
                            seed: int | None = None):
    """
    Generate MAR missingness based on parent values.
    """
    if seed is not None:
        np.random.seed(seed)

    X_m = X_complete.copy()
    n, _ = X_complete.shape

    for m, prt in zip(ms, prt_ms):

        # Random threshold as in R: runif(0.1, 0.7)
        bottom_p = np.random.uniform(0.1, 0.7)
        threshold = norm.ppf(bottom_p)

        # Parent-based indicator
        ind = X_complete[:, prt] < threshold

        # Bernoulli sampling
        h_x = np.random.rand(n) < p_missing_h
        l_x = np.random.rand(n) < p_missing_l

        r = l_x.copy()
        r[ind] = h_x[ind]

        X_m[r, m] = np.nan

    return X_m


# ============================================================
# 3. Generate MCAR reference data
#    Corresponds to data_ref in R
# ============================================================

def generate_mcar_reference(X_complete: np.ndarray,
                            X_mar: np.ndarray,
                            ms: list[int],
                            seed: int | None = None):
    """
    Generate MCAR data with the same number of missing values
    per variable as MAR, but with randomly permuted rows.
    """
    if seed is not None:
        np.random.seed(seed)

    X_mcar = X_complete.copy()
    n = X_complete.shape[0]

    for m in ms:
        mar_mask = np.isnan(X_mar[:, m])
        num_missing = np.sum(mar_mask)

        permuted_idx = np.random.permutation(n)[:num_missing]
        X_mcar[permuted_idx, m] = np.nan

    return X_mcar
