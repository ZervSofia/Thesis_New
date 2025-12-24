import numpy as np
import networkx as nx
from scipy.stats import multivariate_normal

# ============================================================
# 1. Random DAG generation (paper-faithful)
# ============================================================

def random_dag(n_nodes: int,
               p_edge: float | None = None,
               seed: int | None = None):
    """
    Generate a random DAG with expected average degree ≈ 2,
    matching R's randomDAG(num_var, 2/(num_var-1)).
    """
    if seed is not None:
        np.random.seed(seed)

    if p_edge is None:
        p_edge = 2 / (n_nodes - 1)

    adj = np.zeros((n_nodes, n_nodes), dtype=int)

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if np.random.rand() < p_edge:
                adj[i, j] = 1

    G = nx.DiGraph(adj)
    return G, adj


# ============================================================
# 2. Assign random edge weights (linear Gaussian SEM)
# ============================================================

def weighted_adj_matrix(adj: np.ndarray,
                        weight_low: float = -1.0,
                        weight_high: float = 1.0,
                        seed: int | None = None):
    """
    Assign random weights to each directed edge in the DAG.
    """
    if seed is not None:
        np.random.seed(seed)

    W = adj.astype(float)
    edge_indices = np.where(adj == 1)

    W[edge_indices] = np.random.uniform(
        weight_low,
        weight_high,
        size=len(edge_indices[0])
    )

    return W


# ============================================================
# 3. Compute true covariance implied by DAG
# ============================================================

def true_covariance(W: np.ndarray,
                    noise_var: float = 1.0):
    """
    Compute Σ = (I - B)^(-1) Ω (I - B)^(-T),
    where B is the weighted adjacency matrix.
    """
    B = W.T
    n = B.shape[0]

    I = np.eye(n)
    Omega = noise_var * np.eye(n)

    Sigma = np.linalg.inv(I - B) @ Omega @ np.linalg.inv(I - B).T
    return Sigma


# ============================================================
# 4. Sample complete data (joint Gaussian)
# ============================================================

def sample_complete_data(Sigma: np.ndarray,
                          n_samples: int,
                          seed: int | None = None):
    """
    Sample X ~ N(0, Σ), matching rmvnorm in R.
    """
    if seed is not None:
        np.random.seed(seed)

    p = Sigma.shape[0]
    X = multivariate_normal.rvs(
        mean=np.zeros(p),
        cov=Sigma,
        size=n_samples
    )

    return X


# ============================================================
# 5. Collider detection (used for MAR / MNAR)
# ============================================================

def detect_colliders(adj: np.ndarray):
    """
    Return indices of collider nodes (≥2 parents).
    """
    return [j for j in range(adj.shape[1]) if np.sum(adj[:, j]) > 1]


def detect_collider_parents(adj: np.ndarray, colliders: list[int]):
    """
    Return parents of each collider.
    """
    parents = []
    for c in colliders:
        parents.append(list(np.where(adj[:, c] == 1)[0]))
    return parents




# G, adj = random_dag(n_nodes=10, seed=1)
# W = weighted_adj_matrix(adj, seed=1)
# Sigma = true_covariance(W)

# # Check positive definiteness
# np.all(np.linalg.eigvals(Sigma) > 0)
