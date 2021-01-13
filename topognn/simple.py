#!/usr/bin/env python
"""Simple test."""
import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import random_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader, Batch, Data

from topognn import DATA_DIR

from topognn.topo_utils import batch_persistence_routine, persistence_routine, batch_persistence_routine_old

import sys

import argparse

GPU_AVAILABLE = torch.cuda.is_available() and torch.cuda.device_count() > 0


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


class TopologyLayer(torch.nn.Module):
    """Topological Aggregation Layer."""
    
    def __init__(self, features_in, features_out, num_filtrations, num_coord_funs):
        super().__init__()

        self.features_in = features_in
        self.features_out = features_out

        self.num_filtrations = num_filtrations
        self.num_coord_funs = num_coord_funs
        self.filtration = torch.nn.Linear(self.features_in, self.num_filtrations)

        self.coord_fun_c = torch.nn.Parameter(
            torch.randn(self.num_coord_funs, 2),
            requires_grad = True
        )

        self.coord_fun_r = torch.nn.Parameter(
            torch.randn(self.num_coord_funs),
            requires_grad = True
        )
        self.out = torch.nn.Linear(
            self.features_in + self.num_filtrations * self.num_coord_funs, features_out)

    def compute_persistence(self,x,batch):
        """
        Returns the persistence pairs as a list of tensors with shape [X.shape[0],2].
        The lenght of the list is the number of filtrations.
        """ 
        filtered_v_ = self.filtration(x)
        persistences = [ batch_persistence_routine(filtered_v_[:,f_idx],batch) for f_idx in range(self.num_filtrations) ]

        return persistences

    def compute_coord_fun(self,persistence):
        """
        Input : persistence [N_points,2]
        Output : coord_fun mean-aggregated [self.num_coord_fun]
        """

        l1_norm = torch.norm(persistence.unsqueeze(1).repeat(1,self.num_coord_funs,1)-self.coord_fun_c,p = 1, dim = -1)
        coord_activation = (1/(1+l1_norm)) - (1/(1+torch.abs(l1_norm-torch.abs(self.coord_fun_r))))
        #reduced_coord_activation = coord_activation.sum(0)

        return coord_activation


    def compute_coord_activations(self,persistences,batch):

        """
        Return the coordinate functions activations pooled by graph.
        Output dims : list of length number of filtrations with elements : [N_graphs in batch, number fo coordinate functions]
        """
        coord_activations = [self.compute_coord_fun(persistence) for persistence in persistences]

        return torch.cat(coord_activations,1)


    def forward(self,x, batch):

        persistences = self.compute_persistence(x,batch)

        coord_activations = self.compute_coord_activations(persistences,batch)

        concat_activations = torch.cat((x,coord_activations),1)
        
        out_activations = self.out(concat_activations)

        return out_activations


class FiltrationGCNModel(pl.LightningModule):
    """
    GCN Model with a Graph Filtration Readout function.
    """

    def __init__(self, hidden_dim, num_node_features, num_classes, num_filtrations, num_coord_funs):
        """
        num_filtrations = number of filtration functions
        num_coord_funs = number of different coordinate function
        """
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.topo1 = TopologyLayer(hidden_dim, hidden_dim, num_filtrations = num_filtrations,
                                    num_coord_funs = num_coord_funs )
        self.conv3 = GCNConv(hidden_dim, num_classes)

        self.loss = torch.nn.CrossEntropyLoss()

        self.num_filtrations = num_filtrations
        self.num_coord_funs = num_coord_funs

        self.accuracy = pl.metrics.Accuracy()
        self.accuracy_val = pl.metrics.Accuracy()


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

       
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        X = F.relu(x)
        x = F.dropout(x, training = self.training)
       
        x = self.topo1(x,data)
        x = self.conv3(x,edge_index)

        x = global_mean_pool(x, data.batch)

        return x 

    def training_step(self, batch, batch_idx):
        y = batch.y
        y_hat = self(batch)

        loss = self.loss(y_hat,y)

        self.log("train_loss_step",loss)
        self.log("train_acc_step",self.accuracy(y_hat,y))
        return loss

    def training_epoch_end(self,outs):
        self.log("train_acc_epoch",self.accuracy.compute())

    def validation_step(self,batch,batch_idx):
        y = batch.y
        y_hat = self(batch)

        loss = self.loss(y_hat,y)
        self.log("validation_loss",loss)

        self.log("val_acc_step",self.accuracy_val(y_hat,y))
        
        return {"predictions" : y_hat.detach().cpu(), 
                "labels" : y.detach().cpu()}

    def validation_epoch_end(self,outputs):
        self.log("val_acc_epoch", self.accuracy_val.compute())
            

class GCNModel(pl.LightningModule):
    def __init__(self, hidden_dim, num_node_features, num_classes):
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, num_classes)
        self.loss = torch.nn.CrossEntropyLoss()

        self.accuracy = pl.metrics.Accuracy()
        self.accuracy_val = pl.metrics.Accuracy()
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        x = F.relu(x)
        x = F.dropout(x, training = self.training)
       
        x = self.conv3(x,edge_index)

        x = global_mean_pool(x, data.batch)
      
        return x

    def training_step(self, batch, batch_idx):
        y = batch.y
        y_hat = self(batch)

        loss = self.loss(y_hat,y)

        self.log("train_loss_step",loss)
        self.log("train_acc_step",self.accuracy(y_hat,y))
        return loss

    def training_epoch_end(self,outs):
        self.log("train_acc_epoch",self.accuracy.compute())

    def validation_step(self,batch,batch_idx):
        y = batch.y
        y_hat = self(batch)

        loss = self.loss(y_hat,y)
        self.log("validation_loss",loss)

        self.log("val_acc_step",self.accuracy_val(y_hat,y))
        
        return {"predictions" : y_hat.detach().cpu(), 
                "labels" : y.detach().cpu()}

    def validation_epoch_end(self,outputs):
        self.log("val_acc_epoch", self.accuracy_val.compute())


def main(args):
    
    model_type = args.type

    wandb_logger = WandbLogger(name = f"Attempt_{model_type}",project = "topo_gnn",entity = "topo_gnn")
    #wandb_logger = WandbLogger(name = f"Attempt_{model_type}",project = "TopoGNN",entity="edebrouwer")

    trainer = pl.Trainer(
        gpus=-1 if GPU_AVAILABLE else None,
        logger = wandb_logger
    )

    data = TUGraphDataset('ENZYMES', batch_size=32)
    data.prepare_data()

    if model_type=="GCN":
        model = GCNModel(hidden_dim=32, num_node_features=data.node_attributes,
                     num_classes=data.num_classes)
    elif model_type=="TopoGNN":
        model = FiltrationGCNModel(hidden_dim=32, num_node_features=data.node_attributes,
                     num_classes=data.num_classes, num_filtrations = 2, num_coord_funs = 10 )
    else:
        raise("Model not found")

    trainer.fit(model, datamodule=data)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Topo GNN")
    parser.add_argument("--type",type=str,default = "TopoGNN")

    args = parser.parse_args()

    main(args)

    ##--- Test that the filtration is correct.- Comparing against figure 2 of the graph filtration paper ##
    #g = ig.Graph([(0,3),(2,3),(3,4),(4,1)],edge_attrs={"filtration":[3,3,4,4]})
    #g.vs["filtration"] = [1,1,2,3,4]
    #persistence,_ = calculate_persistence_diagrams(
    #        g, vertex_attribute='filtration', edge_attribute='filtration')#, order = "superlevel")
    #print(persistence._pairs)

    batch = Batch() 
    batch.edge_index = torch.tensor([[0,2,3,4],[3,3,4,1]])
    filtered_v = torch.tensor([1.,1.,2.,3.,4.])
    persistence = persistence_routine(filtered_v, batch)
    print(persistence_routine(filtered_v, batch))

