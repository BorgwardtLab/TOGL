#!/usr/bin/env python
"""Sandbox for testing out stuff"""
import os

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import topognn.models as models
import topognn.data_utils as topodata

import sys

import argparse

GPU_AVAILABLE = torch.cuda.is_available() and torch.cuda.device_count() > 0



def main(args):
    
    model_type = args.type

    wandb_logger = WandbLogger(name = f"Another_test_{model_type}",project = "topo_gnn_sandbox",entity = "edebrouwer")
    #wandb_logger = WandbLogger(name = f"Attempt_{model_type}",project = "TopoGNN",entity="edebrouwer")

    trainer = pl.Trainer(
        gpus=-1 if GPU_AVAILABLE else None,
        logger = wandb_logger,
        log_every_n_steps = 5
    )

    data = topodata.TUGraphDataset('ENZYMES', batch_size=32)
    data.prepare_data()

    num_coord_funs = { "Triangle_transform":args.num_coord_funs,
            "Gaussian_transform":args.num_coord_funs,
            "Line_transform":args.num_coord_funs,
            "RationalHat_transform":args.num_coord_funs
            }

    num_coord_funs1 = { "Triangle_transform":args.num_coord_funs1,
            "Gaussian_transform":args.num_coord_funs1,
            "Line_transform":args.num_coord_funs1,
            "RationalHat_transform":args.num_coord_funs1
            }
    
    if model_type=="GCN":
        model = models.GCNModel(hidden_dim=args.hidden_dim, num_node_features=data.node_attributes,
                     num_classes=data.num_classes)
    elif model_type=="TopoGNN":
        model = models.FiltrationGCNModel(hidden_dim= args.hidden_dim,
                filtration_hidden = args.filtration_hidden,
                num_node_features=data.node_attributes,
                     num_classes=data.num_classes,
                     num_filtrations = args.num_filtrations,
                     num_coord_funs = num_coord_funs,
                     dim1 = args.dim1,
                     num_coord_funs1 = num_coord_funs1,
                     lr= args.lr,
                     dropout_p = args.dropout_p )
    else:
        raise("Model not found")

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Topo GNN")
    parser.add_argument("--type",type=str,default = "TopoGNN")
    parser.add_argument("--dim1",action="store_true")
    parser.add_argument("--filtration_hidden",type = int, default = 15)
    parser.add_argument("--num_filtrations",type = int, default = 4)
    parser.add_argument("--hidden_dim",type=int, default = 34)
    parser.add_argument("--num_filtration",type=int, default = 2)
    parser.add_argument("--num_coord_funs",type=int, default = 3)
    parser.add_argument("--num_coord_funs1",type=int, default = 3)
    parser.add_argument("--lr",type=int, default = 0.005)
    parser.add_argument("--dropout_p",type=int, default = 0.5)    

    args = parser.parse_args()

    main(args)

    ##--- Test that the filtration is correct.- Comparing against figure 2 of the graph filtration paper ##
    #g = ig.Graph([(0,3),(2,3),(3,4),(4,1)],edge_attrs={"filtration":[3,3,4,4]})
    #g.vs["filtration"] = [1,1,2,3,4]
    #persistence,_ = calculate_persistence_diagrams(
    #        g, vertex_attribute='filtration', edge_attribute='filtration')#, order = "superlevel")
    #print(persistence._pairs)

    #from torch_geometric.data import Batch
    #from topognn.topo_utils import persistence_routine
    #batch = Batch() 
    #batch.edge_index = torch.tensor([[0,0,2,3,4],[2,3,3,4,1]])
    #filtered_v = torch.tensor([1.,1.,2.,3.,4.])
    #persistence = persistence_routine(filtered_v, batch, cycles = True)
    
    #print(persistence)
