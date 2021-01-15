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

    wandb_logger = WandbLogger(name = f"Attempt_2_layers_dim1_{model_type}",project = "topo_gnn",entity = "topo_gnn")
    #wandb_logger = WandbLogger(name = f"Attempt_{model_type}",project = "TopoGNN",entity="edebrouwer")

    trainer = pl.Trainer(
        gpus=-1 if GPU_AVAILABLE else None,
        logger = wandb_logger,
        log_every_n_steps = 5,
        max_epochs = 2000
    )

    data = topodata.TUGraphDataset('ENZYMES', batch_size=32)
    data.prepare_data()

    num_coord_funs = { "Triangle_transform":3,
            "Gaussian_transform":3,
            "Line_transform":3,
            "RationalHat_transform":3
            }

    num_coord_funs1 = { "Triangle_transform":2,
            "Gaussian_transform":2,
            "Line_transform":2,
            "RationalHat_transform":2
            }
    
    if model_type=="GCN":
        model = models.GCNModel(hidden_dim=32, num_node_features=data.node_attributes,
                     num_classes=data.num_classes, lr = 0.005)
    elif model_type=="TopoGNN":
        model = models.FiltrationGCNModel(hidden_dim=32, filtration_hidden = 10, num_node_features=data.node_attributes,
                     num_classes=data.num_classes, num_filtrations = 2, num_coord_funs = num_coord_funs, dim1 = True, num_coord_funs1 = num_coord_funs1, lr = 0.005 )

    else:
        raise("Model not found")

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Topo GNN")
    parser.add_argument("--type",type=str,default = "TopoGNN")

    args = parser.parse_args()

    main(args)



