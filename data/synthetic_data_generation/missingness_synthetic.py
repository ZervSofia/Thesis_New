import numpy as np
from scipy.stats import norm


# Choose missingness indicators (MAR)
# Corresponds to create_mar_ind() in R

import numpy as np

def create_mar_ind(colliders,
                   collider_parents,
                   num_var: int,
                   num_extra_e: int = 3,
                   num_m: int = 6,
                   seed: int | None = None):
    """
    Translation of create_mar_ind().

    Returns
    -------
    ms : list[int]
        Indices of variables with missing values.
    prt_ms : list[int]
        Indices of parents of the missingness indicators.
    """
    if seed is not None:
        np.random.seed(seed)

    ms = []
    prt_ms = []

    
    # collect all collider-parent pairs 
    for cl, parents in zip(colliders, collider_parents):
        for p in parents:
            if (p not in ms) and (p not in prt_ms):
                ms.append(p)
                prt_ms.append(cl)

    
    # keep only num_extra_e collider-based pairs (randomly)
    if len(ms) > num_extra_e:
        idx = np.random.permutation(len(ms))[:num_extra_e]
        ms = [ms[i] for i in idx]
        prt_ms = [prt_ms[i] for i in idx]

    
    # add more parents (not collider, not in ms/prt_ms)
    left_ind_prt = list(set(range(num_var)) - set(ms) - set(prt_ms))
    np.random.shuffle(left_ind_prt)

    end_for = num_m - len(ms)

    # Append parents
    countp = len(prt_ms)
    for i in range(end_for):
        prt_ms.append(left_ind_prt[i])


    # Step 4: add more missingness indicators 
    
    left_ind_m = list(set(range(num_var)) - set(ms) - set(prt_ms))
    np.random.shuffle(left_ind_m)

    # for i in range(end_for):
    #     ms.append(left_ind_m[i])

    # return ms, prt_ms
    
    for i in range(min(end_for, len(left_ind_m))):
        ms.append(left_ind_m[i])

    return ms, prt_ms


# Generate MAR missing values
# Corresponds to generate_missing_values() + mis_cal_ind()

def generate_missing_values(X_complete: np.ndarray,
                            ms: list[int],
                            prt_ms: list[int],
                            p_missing_h: float = 0.9,
                            p_missing_l: float = 0.1,
                            seed: int | None = None):
    """
    Generate MAR missingness based on parent values.
    """
    if seed is not None:
        np.random.seed(seed)

    X_m = X_complete.copy()
    n, _ = X_complete.shape

    for m, prt in zip(ms, prt_ms):

        
        bottom_p = np.random.uniform(0.1, 0.7)
        threshold = norm.ppf(bottom_p)

        
        ind = X_complete[:, prt] < threshold

        
        h_x = np.random.rand(n) < p_missing_h
        l_x = np.random.rand(n) < p_missing_l

        r = l_x.copy()
        r[ind] = h_x[ind]

        X_m[r, m] = np.nan

    return X_m


# Generate MCAR reference data

def generate_mcar_reference(X_complete: np.ndarray,
                            X_mar: np.ndarray,
                            ms: list[int],
                            seed: int | None = None):
    """
    translation of MCAR generation logic.
    For each missingness indicator m:
        - take the MAR mask r
        - randomly permute r
        - apply it to MCAR data
    """
    if seed is not None:
        np.random.seed(seed)

    X_mcar = X_complete.copy()
    n = X_complete.shape[0]

    for m in ms:
        # MAR mask for this variable
        r = np.isnan(X_mar[:, m])

        # Permute the mask
        permuted_mask = np.random.permutation(r)

        # Apply MCAR missingness
        X_mcar[permuted_mask, m] = np.nan

    return X_mcar





def create_mnar_ind(colliders,
                    collider_parents,
                    num_var: int,
                    num_extra_e: int = 3,
                    num_m: int = 6,
                    seed: int | None = None):
    """
    translation of the R create_mnar_ind() function.

    MNAR constraints:
    - Start from MAR-like collider-based structure
    - Limit to num_extra_e collider-based missingness indicators
    - Add MNAR indicators whose parents ALSO have missing values
    - No self-masking
    - MNAR parents cannot be colliders
    """

    if seed is not None:
        np.random.seed(seed)

    ms = []
    prt_ms = []

    
    # Collect collider-parent pairs
    for cl, parents in zip(colliders, collider_parents):
        for p in parents:
            if (p not in ms) and (p not in prt_ms):
                ms.append(p)
                prt_ms.append(cl)


    # keep only num_extra_e collider-based pairs
    if len(ms) > num_extra_e:
        idx = np.random.permutation(len(ms))[:num_extra_e]
        ms = [ms[i] for i in idx]
        prt_ms = [prt_ms[i] for i in idx]


    # add MNAR indicators
    remaining = num_m - len(ms)

    # MNAR parents cannot be colliders
    left_ind_prt = list(set(range(num_var)) - set(colliders))
    
    # cannot reuse existing ms or prt_ms
    left_ind_prt = list(set(left_ind_prt) - set(ms) - set(prt_ms))
    np.random.shuffle(left_ind_prt)


    left_ind_m = list(set(range(num_var)) - set(ms) - set(prt_ms))
    np.random.shuffle(left_ind_m)

    # assign MNAR indicators
    countm = len(ms)
    countp = len(prt_ms)

    for i in range(remaining):

        # parent must also be missing MNAR 
        if i < num_extra_e:
            new_m = prt_ms[i]     
        else:
            new_m = left_ind_prt[i]

        # no self-masking
        possible_parents = [v for v in left_ind_prt if v != new_m]
        prt = np.random.choice(possible_parents)

        ms.append(new_m)
        prt_ms.append(prt)

    return ms, prt_ms






def choose_missingness_indicators(mode,
                                  colliders,
                                  collider_parents,
                                  num_var,
                                  num_extra_e=3,
                                  num_m=6,
                                  seed=None):

    if mode == "mar":
        return create_mar_ind(colliders, collider_parents,
                              num_var, num_extra_e, num_m, seed)

    elif mode == "mnar":
        return create_mnar_ind(colliders, collider_parents,
                               num_var, num_extra_e, num_m, seed)

    else:
        raise ValueError(f"Unknown mode: {mode}")
