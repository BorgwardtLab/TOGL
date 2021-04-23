"""Clique planting data set for additional experiments."""

import pytorch_lightning as pl

import torch

import numpy as np
import networkx as nx

from torch_geometric.data import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx


class CliquePlanting(InMemoryDataset):
    """Clique planting data set."""

    def __init__(
        self,
        n_graphs=1000,
        n_vertices=500,
        k=20,
    ):
        """Initialise new variant of clique planting data set.

        Parameters
        ----------
        n_graphs : int
            How many graphs to create.

        n_vertices : int
            Size of graph for planting a clique.

        k : int
            Size of clique. Must be subtly 'compatible' with n, but the
            class will warn if problematic values are being chosen.
        """
        super().__init__()
        self.n_graphs = n_graphs
        self.n_vertices = n_vertices
        self.k = k

    def prepare_data(self):
        """Create data set and store it in memory for subsequent processing."""
        graphs = [self._make_graph() for i in range(self.n_graphs)]
        labels = [y for _, y in graphs]

        data, slices = self.collate([from_networkx(g) for g, _ in graphs])
        data.y = torch.tensor(labels, dtype=torch.long)
        print(data)
        print(data.y)

        raise 'heck'

        n = len(data)

        # Simple splitting; in real-world data sets, this should be made
        # more complex.
        n_train = math.floor(
            (1 - self.val_fraction) * (1 - self.test_fraction) * n
        )
        n_val = math.ceil((self.val_fraction) * (1 - self.test_fraction) * n)
        n_test = n - n_train - n_val

        from torch.utils.data import random_split

        # This already splits the data set; we could also go for
        # a `Subset` here.
        self.train, self.val, self.test = random_split(
            data,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(self.seed)
        )

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

    data = CliquePlanting(n_graphs=10)
    data.prepare_data()

    print(next(data.train_dataloader()))
