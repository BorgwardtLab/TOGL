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
from pyper.utilities import UnionFind

import igraph as ig
import sys

GPU_AVAILABLE = torch.cuda.is_available() and torch.cuda.device_count() > 0

def batch_to_igraph_list(batch: Batch):
    list_of_instances = batch.to_data_list()
    # TODO: this conversion can be done more quickly without an
    # intermediate adjacency matrix representation.
    #adjacency_matrices = [to_dense_adj(instance.edge_index)[0] for instance in
    #                      list_of_instances]
    #graphs = [ig.Graph.Adjacency(m.tolist())
    #          for m in adjacency_matrices]

    graphs = [ig.Graph(zip(instance.edge_index[0].tolist(),instance.edge_index[1].tolist())) for instance in list_of_instances]
    
    return graphs

def persistence_routine(filtered_v_, batch: Batch):
    """
    Pytorch based routine to compute the persistence pairs
    Based on pyper routine.
    Inputs : 
        * filtration values of the vertices
        * batch object that stores the graph structure (could be just the edge_index actually)
    """
    
    # Compute the edge filtrations as the max between the value of the nodes.
    filtered_e_, _ = torch.max(torch.stack((filtered_v_[batch.edge_index[0]],filtered_v_[batch.edge_index[1]])),axis=0)

    filtered_v, v_indices = torch.sort(filtered_v_)
    filtered_e, e_indices = torch.sort(filtered_e_)

    uf = UnionFind(len(v_indices))

    persistence = torch.zeros((len(v_indices),2))

    for edge_index, edge_weight in zip(e_indices,filtered_e):
      
        # nodes connected to this edge
        nodes = batch.edge_index[:,edge_index]
        
        younger = uf.find(nodes[0])
        older = uf.find(nodes[1])

        if younger == older : 
            continue
        elif v_indices[younger] < v_indices[older]:
            younger, older = older, younger
            nodes = torch.flip(nodes,[0])

        persistence[nodes[0],0] = filtered_v_[younger]
        persistence[nodes[0],1] = edge_weight

        uf.merge(nodes[0],nodes[1])

    unpaired_value = filtered_e[-1]

    for root in uf.roots():
        persistence[root,0] = filtered_v_[root]
        persistence[root,1] = unpaired_value

    return persistence

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


class FiltrationGCNModel(pl.LightningModule):
    """
    GCN Model with a Graph Filtration Readout function.
    """

    def __init__(self, hidden_dim, num_node_features, pre_filtration_features, num_classes, num_filtrations, num_coord_funs):
        """
        num_filtrations = number of filtration functions
        num_coord_funs = number of different coordinate function
        """
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, pre_filtration_features)
        self.loss = torch.nn.CrossEntropyLoss()

        self.num_filtrations = num_filtrations
        self.num_coord_funs = num_coord_funs

        self.filtration = torch.nn.Linear(pre_filtration_features,self.num_filtrations)

        self.coord_fun_c = torch.nn.Parameter(torch.randn(self.num_coord_funs,2), requires_grad = True) 
        self.coord_fun_r = torch.nn.Parameter(torch.randn(self.num_coord_funs), requires_grad = True)
        self.out = torch.nn.Linear(self.num_filtrations*self.num_coord_funs,num_classes)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def compute_coord_fun(self,persistence):
        """
        Input : persistence [N_points,2]
        Output : coord_fun mean-aggregated [self.num_coord_fun]
        """
        l1_norm = torch.norm(persistence.unsqueeze(1).repeat(1,self.num_coord_funs,1)-self.coord_fun_c,p = 1, dim = -1)
        coord_activation = (1/(1+l1_norm)) - (1/(1+torch.abs(l1_norm-torch.abs(self.coord_fun_r))))
        reduced_coord_activation = coord_activation.sum(0)

        return reduced_coord_activation

    def compute_persistence(self,x,batch):
        
        filtered_v_ = self.filtration(x)

        persistences = [ persistence_routine(filtered_v_[:,f_idx],batch) for f_idx in range(self.num_filtrations) ]

        return persistences

    def compute_coord_activations(self,persistences):

        coord_activations = torch.cat([self.compute_coord_fun(persistence) for persistence in persistences])
        return coord_activations
       
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        persistences = self.compute_persistence(x,data)
        coord_activations = self.compute_coord_activations(persistences)

        x_out = self.out(coord_activations).unsqueeze(0)

        return x_out 

    def training_step(self, batch, batch_idx):
        y = batch.y
        y_hat = self(batch)
        return self.loss(y_hat, y)


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
    data = TUGraphDataset('ENZYMES', batch_size=1)
    data.prepare_data()
    # Test
    batch = next(data.train_dataloader().__iter__())
    diagrams = persistence_diagrams_from_batch(batch)

    
    model = GCNModel(hidden_dim=32, num_node_features=data.node_attributes,
                     num_classes=data.num_classes)
    trainer.fit(model, datamodule=data)

    #TODO : currenty only working on cpu (check devices)
    assert GPU_AVAILABLE is False 
    filtration_model = FiltrationGCNModel(hidden_dim=32, num_node_features=data.node_attributes, pre_filtration_features = 3,
                     num_classes=data.num_classes, num_filtrations = 2, num_coord_funs = 10 )
    trainer.fit(filtration_model, datamodule=data)



    


    ##--- Test that the filtration is correct.- Comparing against figure 2 of the graph filtration paper ##
    g = ig.Graph([(0,3),(2,3),(3,4),(4,1)],edge_attrs={"filtration":[3,3,4,4]})
    g.vs["filtration"] = [1,1,2,3,4]
    persistence,_ = calculate_persistence_diagrams(
            g, vertex_attribute='filtration', edge_attribute='filtration')#, order = "superlevel")
    print(persistence._pairs)


    batch = Batch()
    batch.edge_index = torch.tensor([[0,2,3,4],[3,3,4,1]])
    filtered_v = torch.tensor([1,1,2,3,4])
    persistence = persistence_routine(filtered_v, batch)
    print(persistence_routine(filtered_v, batch))
