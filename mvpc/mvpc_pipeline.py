"""
mvpc_pipeline.py

High-level orchestration of the MVPC algorithm (Missing Value PC).
This module coordinates:
    - detection of missingness indicator parents
    - corrected skeleton search (MVPC step 2)
    - final PC orientation using causal-learn

The heavy lifting is delegated to:
    missingness.py
    skeleton.py
    ci_tests/
"""

import numpy as np
from causallearn.search.ConstraintBased.PC import pc

from .missingness import detection_prt_m
from .skeleton import skeleton2


class MVPC:
    """
    Main MVPC pipeline class.

    Parameters
    ----------
    indep_test : callable
        Base conditional independence test (e.g., Gaussian CI test).
        This is used for:
            - detecting missingness parents
            - initial PC skeleton
    corr_test : callable
        Corrected CI test (DRW or PermC).
        This is used for:
            - corrected skeleton search (MVPC step 2)
    alpha : float
        Significance threshold for CI tests.
    """

    def __init__(self, indep_test, corr_test, alpha=0.05):
        self.indep_test = indep_test
        self.corr_test = corr_test
        self.alpha = alpha

    def run(self, data):
        
        n, p = data.shape

        # ---------------------------------------------------------
        # Step 1: Detect parents of missingness indicators
        # ---------------------------------------------------------
        prt_m = detection_prt_m(
            data=data,
            indep_test=self.indep_test,
            alpha=self.alpha,
            p=p
        )

        # ---------------------------------------------------------
        # Step 2a: Initial skeleton (undirected)
        # ---------------------------------------------------------
        pc_initial = pc(
            data,
            ci_test=self.indep_test,
            alpha=self.alpha
        )

        # Extract skeleton (symmetrize adjacency)
        G_skel = pc_initial.G
        G_skel = ((G_skel + G_skel.T) > 0).astype(int)

        # Wrap in simple object for skeleton2
        class SimpleSkeleton:
            def __init__(self, G):
                self.G = G

        skel_pre = SimpleSkeleton(G_skel)

        # ---------------------------------------------------------
        # Step 2b: Corrected skeleton
        # ---------------------------------------------------------
        G_corrected, sepset_corrected, pmax_corrected = skeleton2(
            data=data,
            corr_test=self.corr_test,
            alpha=self.alpha,
            skel_pre=skel_pre,
            prt_m=prt_m
        )

        # ---------------------------------------------------------
        # Step 2c: Orientation on corrected skeleton
        # ---------------------------------------------------------
        pc_obj = pc(
            data,
            ci_test=self.corr_test,
            alpha=self.alpha
        )

        # Inject corrected skeleton
        pc_obj.G = G_corrected

        # Run only orientation rules
        pc_obj.orient()

        return pc_obj
