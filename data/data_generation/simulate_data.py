

import numpy as np
import pandas as pd
from data.data_generation.tetrad_graph_to_array import tetrad_graph_to_array

def simulate_data_tetrad_simple(
    n_nodes,
    n_samples,
    avg_degree,
    max_degree,
    seed=None
):
    from java import util
    from edu.cmu.tetrad.algcomparison import simulation, graph

    parameters = util.Parameters()
    parameters.set("numMeasures", n_nodes)
    parameters.set("sampleSize", n_samples)
    parameters.set("avgDegree", avg_degree)
    parameters.set("maxDegree", max_degree)
    parameters.set("numLatents", 0)
    parameters.set("numRuns", 1)
    parameters.set("differentGraphs", True)
    parameters.set("randomizeColumns", False)

    if seed is not None:
        parameters.set("seed", seed)

    G = graph.RandomForward()

    Sim = simulation.GeneralSemSimulation(G)
    Sim.createData(parameters, True)

    tDag = Sim.getTrueGraph(0)
    tData = Sim.getDataModel(0)

    dag = tetrad_graph_to_array(tDag)
    data_ = np.array([
        [tData.getDouble(i, j) for j in range(tData.getNumColumns())]
        for i in range(tData.getNumRows())
    ])

    var_names = [f'V{i+1}' for i in range(n_nodes)]

    dag_pd = pd.DataFrame(dag, columns=var_names, index=var_names)
    data_pd = pd.DataFrame(data_, columns=var_names)

    return dag_pd, data_pd
