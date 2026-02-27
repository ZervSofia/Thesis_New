"""
thin wrapper around MVPC that allows overriding the missingness-parent
structure (prt_m) with oracle information from synthetic experiments.


"""

import numpy as np
from .mvpc_pipeline import _pc_skeleton_initial
from .skeleton import skeleton2


class MVPC_Oracle:
    def __init__(self, indep_test, corr_test, alpha=0.05):
        self.indep_test = indep_test
        self.corr_test = corr_test
        self.alpha = alpha

    def run(self, data, prt_m):
        """
        Run MVPC using oracle missingness-parent structure.

        Parameters
        ----------
        data : np.ndarray
            Data matrix with missing values.
        prt_m : dict
            Oracle missingness-parent structure:
                {"m": [...], "prt": {m: [parents...]}}
        """
        n, p = data.shape

        # initial skeleton 
        G_initial, sepset_initial = _pc_skeleton_initial(
            data=data,
            indep_test=self.indep_test,
            alpha=self.alpha
        )

        class SimpleSkeleton:
            def __init__(self, G):
                self.G = G

        skel_pre = SimpleSkeleton(G_initial)

        # corrected skeleton using oracle prt_m
        G_corrected, sepset_corrected, pmax_corrected = skeleton2(
            data=data,
            corr_test=self.corr_test,
            alpha=self.alpha,
            skel_pre=skel_pre,
            prt_m=prt_m
        )

        return {
            "G_initial": G_initial,
            "G_corrected": G_corrected.astype(int),
            "sepset_initial": sepset_initial,
            "sepset_corrected": sepset_corrected,
            "pmax_corrected": pmax_corrected,
            "prt_m": prt_m,
        }
