"""
Core utilities for MVPC
- test_wise_deletion
- perm
- get_prt_i
- get_prt_m_xys
- is_in_prt_m
- cond_PermC
- common_neighbor
"""

import numpy as np



PERMC_DIAG = {
    "total_calls": 0,
    "used": 0,       
    "fallback": 0,   
}


# Test-wise deletion (R: test_wise_deletion)

def test_wise_deletion(var_ind, data):
    # var_ind: list of column indices
    mask = np.ones(data.shape[0], dtype=bool)
    for v in var_ind:
        col = data[:, v]
        if np.isnan(col).any():
            mask &= ~np.isnan(col)
    return data[mask, :]


def perm(W, data):
    # test-wise deletion on W
    data_tw = test_wise_deletion(W, data)
    n = data_tw.shape[0]
    idx = np.random.permutation(n)
    data_perm = data_tw[idx, :]
    return data_perm[:, W]



def get_prt_i(ind_ri, suffstat):
    prt_m = suffstat["prt_m"]
    m_list = prt_m["m"]
    prt_dict = prt_m["prt"]

    if ind_ri not in m_list:
        return []
    return prt_dict.get(ind_ri, [])


def is_in_prt_m(i, prt_m):

    return i in prt_m["m"]


def get_prt_m_xys(ind, suffstat):
    prt_m = suffstat["prt_m"]
    w = []
    for i in ind:
        if is_in_prt_m(i, prt_m):
            prt_i = get_prt_i(i, suffstat)
            w.extend(prt_i)
    return w



# cond.PermC and common.neighbor

def common_neighbor(x, y, skel):
    """
    skel is an adjacency matrix (0/1 or bool), shape (p, p).
    Returns True if x and y share at least one common neighbor.
    """
    skel_mat = np.asarray(skel)
    return np.any((skel_mat[:, x] == 1) & (skel_mat[:, y] == 1))




def cond_PermC(x, y, S, suffstat):
    """
    translation of R cond.PermC
    
    """

    PERMC_DIAG["total_calls"] += 1

    ind = [x, y] + list(S)

    # If no skeleton provided, always do correction
    if "skel" not in suffstat:
        PERMC_DIAG["used"] += 1
        return True

    prt_m = suffstat["prt_m"]
    m_list = set(prt_m["m"])

    # xyS have missingness indicators with parents
    if len(set(ind) & m_list) == 0:
        PERMC_DIAG["fallback"] += 1
        return False

    # x and y have a common neighbor in the current skeleton
    skel = suffstat["skel"]
    if common_neighbor(x, y, skel):
        PERMC_DIAG["used"] += 1
        return True
    else:
        PERMC_DIAG["fallback"] += 1
        return False
