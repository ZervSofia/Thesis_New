import numpy as np
import pandas as pd
import networkx as nx

def simulate_linear_scm(
    n_nodes=10,
    n_samples=1000,
    avg_degree=2,
    noise_scale=1.0,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)

    p_edge = avg_degree / (n_nodes - 1)

    # Create random DAG
    while True:
        G = nx.gnp_random_graph(n_nodes, p_edge, directed=True)
        # Enforce acyclicity by keeping edges i->j only if i < j
        G = nx.DiGraph((i, j) for i, j in G.edges if i < j)
        if len(G.nodes) == n_nodes:  # ensure all nodes are present
            break

    weights = {
        (i, j): np.random.uniform(-2, 2)
        for i, j in G.edges
    }

    X = np.zeros((n_samples, n_nodes))
    order = list(nx.topological_sort(G))

    for j in order:
        parents = list(G.predecessors(j))
        X[:, j] = sum(
            weights[(i, j)] * X[:, i] for i in parents
        ) + np.random.normal(scale=noise_scale, size=n_samples)

    var_names = [f"X{i+1}" for i in range(n_nodes)]
    data = pd.DataFrame(X, columns=var_names)

    # Create full adjacency matrix including isolated nodes
    dag_array = np.zeros((n_nodes, n_nodes), dtype=int)
    for i, j in G.edges:
        dag_array[i, j] = 1
    dag = pd.DataFrame(dag_array, columns=var_names, index=var_names)

    return dag, data
