import numpy as np
from scipy.stats import norm
from numpy.linalg import pinv
from causal_discovery.ci_tests.base import CITest


class ListwiseCITest(CITest):

    def is_independent(self, X, i, j, S, alpha):
        # drop all rows with any missing value
        mask = ~np.isnan(X).any(axis=1)
        Xc = X[mask]

        n = Xc.shape[0]
        if n <= len(S) + 3:
            return False  # not enough data

        if len(S) == 0:
            r = np.corrcoef(Xc[:, i], Xc[:, j])[0, 1]
        else:
            idx = [i, j] + S
            cov = np.cov(Xc[:, idx], rowvar=False)
            prec = pinv(cov)
            r = -prec[0, 1] / np.sqrt(prec[0, 0] * prec[1, 1])

        z = 0.5 * np.log((1 + r) / (1 - r))
        test_stat = np.sqrt(n - len(S) - 3) * abs(z)
        p_value = 2 * (1 - norm.cdf(test_stat))

        return p_value > alpha
