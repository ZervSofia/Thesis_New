import numpy as np
import pandas as pd
import os

# -------------------------------------------------------
# SETTINGS
# -------------------------------------------------------
N_SAMPLES = 2000
N_VARS = 10
OUTPUT_DIR = "data/processed"

np.random.seed(42)


# -------------------------------------------------------
# STEP 1 — Generate a random DAG (acyclic adjacency matrix)
# -------------------------------------------------------
def random_dag(num_vars, edge_prob=0.25):
    """Generate a random DAG by sampling upper triangular adjacency."""
    A = np.zeros((num_vars, num_vars))
    for i in range(num_vars):
        for j in range(i+1, num_vars):  # j > i ensures no cycles
            if np.random.rand() < edge_prob:
                A[i, j] = 1
    return A


# -------------------------------------------------------
# STEP 2 — Simulate linear SEM
# -------------------------------------------------------
def simulate_linear_sem(A, n_samples):
    num_vars = A.shape[0]
    X = np.zeros((n_samples, num_vars))

    # Random weights for each edge
    weights = np.random.uniform(0.3, 1.0, size=A.shape) * A

    for i in range(num_vars):
        parents = np.where(A[:, i] == 1)[0]

        if len(parents) == 0:
            # Exogenous variable
            X[:, i] = np.random.normal(0, 1, n_samples)
        else:
            X[:, i] = (
                X[:, parents].dot(weights[parents, i]) +
                np.random.normal(0, 1, n_samples)
            )

    return X, weights


# -------------------------------------------------------
# STEP 3 — Inject missingness (optional)
# -------------------------------------------------------
def inject_missingness(data, mechanism="MCAR", rate=0.2):
    data = data.copy()

    if mechanism == "MCAR":
        mask = np.random.rand(*data.shape) < rate
        data[mask] = np.nan

    elif mechanism == "MAR":
        # Missingness depends on a different observed variable
        for j in range(data.shape[1]):
            prob = (data.iloc[:, (j+1) % data.shape[1]] > 0).astype(float) * rate
            mask = np.random.rand(data.shape[0]) < prob
            data.loc[mask, data.columns[j]] = np.nan

    elif mechanism == "MNAR":
        # Missingness depends on the variable's own value
        for j in range(data.shape[1]):
            prob = (data.iloc[:, j] > data.iloc[:, j].median()).astype(float) * rate
            mask = np.random.rand(data.shape[0]) < prob
            data.loc[mask, data.columns[j]] = np.nan

    return data


# -------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------
if __name__ == "__main__":

    print("Generating synthetic causal dataset...")

    # 1. DAG
    A = random_dag(N_VARS, edge_prob=0.25)

    # 2. Simulate SEM
    X, weights = simulate_linear_sem(A, N_SAMPLES)

    # 3. Convert to DataFrame
    df = pd.DataFrame(X, columns=[f"X{i}" for i in range(1, N_VARS+1)])

    # 4. Inject missingness example
    df_mcar = inject_missingness(df, mechanism="MCAR", rate=0.1)

    # Save outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df.to_csv(os.path.join(OUTPUT_DIR, "synthetic_causal_clean.csv"), index=False)
    df_mcar.to_csv(os.path.join(OUTPUT_DIR, "synthetic_causal_mcar.csv"), index=False)
    np.save(os.path.join(OUTPUT_DIR, "synthetic_causal_adj_matrix.npy"), A)

    print("Saved:")
    print(" - synthetic_causal_clean.csv")
    print(" - synthetic_causal_mcar.csv")
    print(" - synthetic_causal_adj_matrix.npy (ground truth DAG)")
