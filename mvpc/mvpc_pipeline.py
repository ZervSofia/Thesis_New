"""
mvpc_pipeline.py

High-level orchestration of the MVPC algorithm (Missing Value PC).

Pipeline:
    1. Detect parents of missingness indicators (Step 1).
    2. Build an initial PC-style skeleton using the base CI test.
    3. Correct the skeleton using the corrected CI test (DRW / PermC).
    4. (Optional) Orient edges – not implemented here, returns skeleton.
"""

import numpy as np
from itertools import combinations

from .missingness import detection_prt_m
from .skeleton import skeleton2


def _pc_skeleton_initial(data, indep_test, alpha):
    """
    Minimal PC-style skeleton search using the *base* CI test.

    This is the analogue of pcalg::skeleton() in R, but simplified:
        - undirected graph
        - no fixedGaps/fixedEdges
        - no conservative/maj.rule variants
    """
    n, p = data.shape

    # fully connected undirected graph (no self-loops)
    G = np.ones((p, p), dtype=bool)
    np.fill_diagonal(G, False)

    # separation sets (optional, kept for completeness)
    sepset = [[None for _ in range(p)] for _ in range(p)]

    ord_size = 0
    done = False

    while not done and G.any():
        done = True
        edges = np.argwhere(G)

        for x, y in edges:
            if x >= y:
                continue  # avoid double testing

            # neighbors of x excluding y
            neighbors = [k for k in range(p) if G[k, x] and k != y]

            if len(neighbors) < ord_size:
                continue

            if len(neighbors) > ord_size:
                done = False

            independent = False

            for S in combinations(neighbors, ord_size):
                suffstat = {"data": data}
                pval = indep_test(x, y, S, suffstat)

                if pval >= alpha:
                    G[x, y] = G[y, x] = False
                    sepset[x][y] = list(S)
                    sepset[y][x] = list(S)
                    independent = True
                    break

            if independent:
                continue

        ord_size += 1

    return G.astype(int), sepset


class MVPC:
    """
    Main MVPC pipeline class.

    Parameters
    ----------
    indep_test : callable
        Base CI test: indep_test(x, y, S, suffstat) -> p-value.
        Used for:
            - detecting missingness parents
            - initial skeleton search
    corr_test : callable
        Corrected CI test (DRW or PermC):
            corr_test(x, y, S, suffstat) -> p-value.
        Used for:
            - corrected skeleton search (MVPC step 2b)
    alpha : float
        Significance threshold for CI tests.
    """

    def __init__(self, indep_test, corr_test, alpha=0.05):
        self.indep_test = indep_test
        self.corr_test = corr_test
        self.alpha = alpha

    def run(self, data):
        """
        Run the MVPC pipeline on a data matrix with missing values (NaNs).

        Returns
        -------
        dict
            {
                "G_initial": initial skeleton (p x p, 0/1),
                "G_corrected": corrected skeleton (p x p, 0/1),
                "sepset_initial": separation sets from initial skeleton,
                "sepset_corrected": separation sets from corrected skeleton,
                "pmax_corrected": max p-values from corrected skeleton search,
                "prt_m": missingness-parent structure
            }
        """
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
        # Step 2a: Initial skeleton (undirected, base CI test)
        # ---------------------------------------------------------
        G_initial, sepset_initial = _pc_skeleton_initial(
            data=data,
            indep_test=self.indep_test,
            alpha=self.alpha
        )

        # Wrap in simple object for skeleton2 (to mimic skel_pre)
        class SimpleSkeleton:
            def __init__(self, G):
                self.G = G

        skel_pre = SimpleSkeleton(G_initial)

        # ---------------------------------------------------------
        # Step 2b: Corrected skeleton (MVPC correction step)
        # ---------------------------------------------------------
        G_corrected, sepset_corrected, pmax_corrected = skeleton2(
            data=data,
            corr_test=self.corr_test,
            alpha=self.alpha,
            skel_pre=skel_pre,
            prt_m=prt_m
        )

        # ---------------------------------------------------------
        # Step 2c: Orientation (not implemented here)
        # ---------------------------------------------------------
        # In the R code, this is done via udag2pdagRelaxed / pc.cons.intern.
        # You can add an orientation step later if needed.
        # For now, we return the corrected skeleton.

        return {
            "G_initial": G_initial,
            "G_corrected": G_corrected.astype(int),
            "sepset_initial": sepset_initial,
            "sepset_corrected": sepset_corrected,
            "pmax_corrected": pmax_corrected,
            "prt_m": prt_m,
        }
