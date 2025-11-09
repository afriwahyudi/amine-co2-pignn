import numpy as np

def compute_statistics(graphs):
    conc_values = [graph.conc for graph in graphs]
    temp_values = [graph.temp for graph in graphs]
    pco2_values = [graph.pco2 for graph in graphs]

    conc_mean, conc_std = np.mean(conc_values), np.std(conc_values)
    temp_mean, temp_std = np.mean(temp_values), np.std(temp_values)
    pco2_mean, pco2_std = np.mean(pco2_values), np.std(pco2_values)

    return conc_mean, conc_std, temp_mean, temp_std, pco2_mean, pco2_std

def scale_graphs(graphs, conc_mean, conc_std, temp_mean, temp_std, pco2_mean, pco2_std):
    if isinstance(graphs, list):
        for graph in graphs:
            graph.conc = (graph.conc - conc_mean) / conc_std
            graph.temp = (graph.temp - temp_mean) / temp_std
            graph.pco2 = (graph.pco2 - pco2_mean) / pco2_std
        return graphs
    else:
        graphs.conc = (graphs.conc - conc_mean) / conc_std
        graphs.temp = (graphs.temp - temp_mean) / temp_std
        graphs.pco2 = (graphs.pco2 - pco2_mean) / pco2_std
        return graphs