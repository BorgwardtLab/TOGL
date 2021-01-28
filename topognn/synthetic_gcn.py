#!/usr/bin/env python
"""Sandbox for testing out stuff"""
import os

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from topognn import Tasks
import topognn.models as models
import topognn.data_utils as topodata

import sys

import argparse

GPU_AVAILABLE = torch.cuda.is_available() and torch.cuda.device_count() > 0



def main(args):
    
    model_type = args.type
    
    if args.GIN:
        model_name = "GIN"
    else:
        model_name = "GCN"

    wandb_logger = WandbLogger(name = f"{model_name}_{args.dataset}",project = "topo_gnn",entity = "topo_gnn", tags = [args.dataset], log_model = True)
    
    #wandb_logger = WandbLogger(name = f"Attempt_{model_type}",project = "TopoGNN",entity="edebrouwer")

    early_stopping_cb = EarlyStopping(monitor="val_acc", patience=200)
    checkpoint_cb = ModelCheckpoint(
        dirpath=wandb_logger.experiment.dir,
        monitor='val_acc',
        mode='max',
        verbose=True
    )
    
    trainer = pl.Trainer(
        gpus=-1 if GPU_AVAILABLE else None,
        logger = wandb_logger,
        log_every_n_steps = 5,
        max_epochs = args.max_epochs,
        callbacks=[early_stopping_cb, checkpoint_cb]
    )

    if args.dataset=="IMDB-BINARY":
        add_node_degree = True
    else:
        add_node_degree = False
    
    data = topodata.SyntheticDataset(args.dataset, batch_size=32,add_node_degree=add_node_degree, seed = args.seed)
    data.prepare_data()

    model = models.GCNModel(hidden_dim=args.hidden_dim,
                num_node_features=data.node_attributes,
                num_classes=data.num_classes,
                task = Tasks.GRAPH_CLASSIFICATION,
                dropout_p = args.dropout_p,
                lr = args.lr, GIN = args.GIN)


    trainer.fit(model, datamodule=data)

    checkpoint_path = checkpoint_cb.best_model_path
    trainer2 = pl.Trainer(logger=False)
    model = models.GCNModel.load_from_checkpoint(
        checkpoint_path)

    val_results = trainer2.test(
        model,
        test_dataloaders=data.val_dataloader()
    )[0]
    
    val_results = {
        name.replace('test', 'best_val'): value
        for name, value in val_results.items()
    }
    test_results = trainer2.test(
        model,
        test_dataloaders=data.test_dataloader()
    )[0]

    for name, value in {**val_results, **test_results}.items():
        wandb_logger.experiment.summary[name] = value


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Topo GNN")
    parser.add_argument("--type",type=str,default = "GCN")
    parser.add_argument("--num_filtrations",type = int, default = 2)
    parser.add_argument("--hidden_dim",type=int, default = 34)
    parser.add_argument("--lr",type=float, default = 0.005)
    parser.add_argument("--dropout_p",type=float, default = 0.1)
    parser.add_argument("--max_epochs",type=int,default = 1000)
    parser.add_argument("--dataset",type=str, default = "Cycles")
    parser.add_argument("--seed",type=int, default = 42)
    parser.add_argument("--GIN",type=bool, default= False)
    
    args = parser.parse_args()

    main(args)



