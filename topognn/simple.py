#!/usr/bin/env python
"""Simple test."""
import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import random_split

import pytorch_lightning as pl
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader, Batch
from torch_geometric.utils import to_dense_adj

from topognn import DATA_DIR
from pyper.persistent_homology.graphs import calculate_persistence_diagrams

import igraph as ig


GPU_AVAILABLE = torch.cuda.is_available() and torch.cuda.device_count() > 0


def batch_to_igraph_list(batch: Batch):
    list_of_instances = batch.to_data_list()
    # TODO: this conversion can be done more quickly without an
    # intermediate adjacency matrix representation.
    adjacency_matrices = [to_dense_adj(instance.edge_index)[0] for instance in
                          list_of_instances]
    graphs = [ig.Graph.Adjacency(m.tolist())
              for m in adjacency_matrices]

    return graphs


def apply_degree_filtration(graphs):
    for graph in graphs:
        graph.vs['filtration'] = graph.vs.degree()

        for edge in graph.es:
            u, v = edge.source, edge.target
            edge['filtration'] = max(graph.vs[u]['filtration'],
                                     graph.vs[v]['filtration'])

    return graphs


def calculate_batch_persistence_diagrams(graphs):
    persistence_diagrams = [
        calculate_persistence_diagrams(
            graph, vertex_attribute='filtration', edge_attribute='filtration')
        for graph in graphs
    ]
    return persistence_diagrams


def persistence_diagrams_from_batch(batch: Batch):
    graphs = batch_to_igraph_list(batch)
    graphs = apply_degree_filtration(graphs)
    return calculate_batch_persistence_diagrams(graphs)


class TUGraphDataset(pl.LightningDataModule):
    def __init__(self, name, batch_size, use_node_attributes=True,
                 val_fraction=0.1, test_fraction=0.1, seed=42, num_workers=4):
        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.use_node_attributes = use_node_attributes
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed
        self.num_workers = num_workers

    def prepare_data(self):
        dataset = TUDataset(
            root=os.path.join(DATA_DIR, self.name),
            use_node_attr=True,
            cleaned=self.use_node_attributes,
            name=self.name
        )
        self.node_attributes = dataset.num_node_features
        self.num_classes = dataset.num_classes
        n_instances = len(dataset)
        n_train = math.floor(
            (1 - self.val_fraction) * (1 - self.test_fraction) * n_instances)
        n_val = math.ceil(
            (self.val_fraction) * (1 - self.test_fraction) * n_instances)
        n_test = n_instances - n_train - n_val

        self.train, self.val, self.test = random_split(
            dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
        )


class GCNModel(pl.LightningModule):
    def __init__(self, hidden_dim, num_node_features, num_classes):
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.loss = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, data.batch)

        return x

    def training_step(self, batch, batch_idx):
        y = batch.y
        y_hat = self(batch)
        return self.loss(y_hat, y)


if __name__ == "__main__":
    trainer = pl.Trainer(
        gpus=-1 if GPU_AVAILABLE else None,
    )
    data = TUGraphDataset('ENZYMES', batch_size=32)
    data.prepare_data()
    # Test
    batch = next(data.train_dataloader().__iter__())
    diagrams = persistence_diagrams_from_batch(batch)
    import pdb
    pdb.set_trace()
    model = GCNModel(hidden_dim=32, num_node_features=data.node_attributes,
                     num_classes=data.num_classes)
    trainer.fit(model, datamodule=data)
