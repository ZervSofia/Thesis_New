"""
mvpc_utils.py

Core utilities for MVPC, faithful to the original R implementation:
- test_wise_deletion
- perm
- get_prt_i
- get_prt_m_xys
- is_in_prt_m
- cond_PermC
- common_neighbor
"""

import numpy as np


# ---------------------------------------------------------
# Test-wise deletion (R: test_wise_deletion)
# ---------------------------------------------------------
def test_wise_deletion(var_ind, data, return_mask=False):
    """
    Delete rows with any missing value among variables in var_ind.

    If return_mask=True, return the boolean mask instead of the data.
    """
    mask = np.ones(data.shape[0], dtype=bool)
    for v in var_ind:
        mask &= ~np.isnan(data[:, v])

    if return_mask:
        return mask
    return data[mask, :]




# ---------------------------------------------------------
# perm (R: perm)
# ---------------------------------------------------------
def perm(W, data):
    """
    R's perm(W, data):

        data <- test_wise_deletion(W, data)
        len = nrow(data)
        ind_p <- sample(1:len)
        data = data[ind_p, ]
        data_permw <- data[, W]
        data.frame(data_permw)

    Here we return a numpy array of shape (n_twdel, len(W)).
    """
    W = list(W)
    data_tw = test_wise_deletion(W, data)
    n = data_tw.shape[0]
    idx = np.random.permutation(n)
    data_perm = data_tw[idx, :]
    return data_perm[:, W]


# ---------------------------------------------------------
# prt_m helpers: get_prt_i, is_in_prt_m, get_prt_m_xys
# ---------------------------------------------------------
def get_prt_i(ind_ri, suffstat):
    """
    R: get_prt_i

    prt_m in Python is assumed to be:
        suffstat["prt_m"] = {"m": [...], "prt": {m_idx: [parents]}}
    """
    prt_m = suffstat["prt_m"]
    m_list = prt_m["m"]
    prt_dict = prt_m["prt"]

    if ind_ri not in m_list:
        return []
    return prt_dict.get(ind_ri, [])


def is_in_prt_m(i, prt_m):
    """
    R: is.in_prt_m

    prt_m: {"m": [...], "prt": {...}}
    """
    return i in prt_m["m"]


def get_prt_m_xys(ind, suffstat):
    """
    R: get_prt_m_xys

    For each variable in ind, if it has a missingness indicator with parents,
    collect those parents.
    """
    prt_m = suffstat["prt_m"]
    w = []
    for i in ind:
        if is_in_prt_m(i, prt_m):
            prt_i = get_prt_i(i, suffstat)
            w.extend(prt_i)
    return w


# ---------------------------------------------------------
# cond.PermC and common.neighbor
# ---------------------------------------------------------
def common_neighbor(x, y, skel):
    """
    R: common.neighbor

    skel is an adjacency matrix (0/1 or bool), shape (p, p).
    Returns True if x and y share at least one common neighbor.
    """
    # ensure numpy array
    skel_mat = np.asarray(skel)
    return np.any((skel_mat[:, x] == 1) & (skel_mat[:, y] == 1))


def cond_PermC(x, y, S, suffstat):
    """
    R: cond.PermC

    Logic:
      ind <- c(x, y, S)
      cond <- FALSE
      if ("skel" %in% names(suffStat)) {
        if (length(intersect(ind, suffStat$prt_m$m)) > 0 &&
            common.neighbor(x, y, suffStat$skel)) {
          cond <- TRUE
        }
        return(cond)
      } else {
        return(TRUE)
      }
    """
    ind = [x, y] + list(S)

    # If no skeleton provided, always do correction
    if "skel" not in suffstat:
        return True

    prt_m = suffstat["prt_m"]
    m_list = set(prt_m["m"])

    # 1) xyS have missingness indicators with parents
    if len(set(ind) & m_list) == 0:
        return False

    # 2) x and y have a common neighbor in the current skeleton
    skel = suffstat["skel"]
    return common_neighbor(x, y, skel)
