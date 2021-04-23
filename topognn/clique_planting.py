"""Clique planting data set for additional experiments."""

import os
import torch

import numpy as np
import networkx as nx

from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx


class CliquePlanting(InMemoryDataset):
    """Clique planting data set."""

    def __init__(
        self,
        root,
        n_graphs=1000,
        n_vertices=500,
        k=20,
    ):
        """Initialise new variant of clique planting data set.

        Parameters
        ----------
        root : str
            Root directory for storing graphs.

        n_graphs : int
            How many graphs to create.

        n_vertices : int
            Size of graph for planting a clique.

        k : int
            Size of clique. Must be subtly 'compatible' with n, but the
            class will warn if problematic values are being chosen.
        """
        self.n_graphs = n_graphs
        self.n_vertices = n_vertices
        self.k = k

        super().__init__(root)

    @property
    def raw_file_names(self):
        """No raw file names are required."""
        return []

    @property
    def processed_dir(self):
        """Directory to store data in."""
        return os.path.join(
            self.root,
            # Following the other nomenclature
            'CLIQUE_PLANTING',
            'processed'
        )

    @property
    def processed_file_names(self):
        """No raw file names are required."""
        return ['data.pt']

    def process(self):
        """Create data set and store it in memory for subsequent processing."""
        graphs = [self._make_graph() for i in range(self.n_graphs)]
        labels = [y for _, y in graphs]

        data, slices = self.collate([from_networkx(g) for g, _ in graphs])
        data.y = torch.tensor(labels, dtype=torch.long)

        torch.save((data, slices), self.processed_paths[0])
        return data, slices

    def _make_graph(self):
        """Create graph potentially containing a planted clique."""
        G = nx.erdos_renyi_graph(self.n_vertices, p=0.50)
        y = 0

        if np.random.choice([True, False]):
            G = self._plant_clique(G, self.k)
            y = 1

        return G, y

    def _plant_clique(self, G, k):
        """Plant $k$-clique in a given graph G.

        This function chooses a random subset of the vertices of the graph and
        turns them into fully-connected subgraph.
        """
        n = G.number_of_nodes()
        vertices = np.random.choice(np.arange(n), k, replace=False)

        for index, u in enumerate(vertices):
            for v in vertices[index+1:]:
                G.add_edge(u, v)

        return G


if __name__ == '__main__':

    data = CliquePlanting(root='data', n_graphs=10)
