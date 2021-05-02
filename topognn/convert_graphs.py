
import pickle
import torch

import networkx as nx
import numpy as np

import matplotlib.pyplot as plt

with open('persistence_analysis/persistence_batch.pkl', 'rb') as f:
    data = pickle.load(f)
    data = data.to_data_list()

for graph in data:
    edges = graph.edge_index
    edges = edges.transpose(0, 1).numpy()

    G = nx.Graph()
    G.add_edges_from(edges)

    degrees = [d for _, d in G.degree()]

    for k in range(graph.x.numpy().shape[-1]):
        nx.set_node_attributes(
            G,
            {
                i + 1: x for i, x in enumerate(graph.x.numpy()[:, k].tolist())
            },
            f'filtration_{k}'
        )

        print(np.corrcoef(graph.x.numpy()[:, k].tolist(), degrees)[0, 1])

    nx.write_graphml(G, '/tmp/foo.graphml')


    plt.show()
