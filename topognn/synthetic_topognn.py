#!/usr/bin/env python
"""Sandbox for testing out stuff"""
import os

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import topognn.models as models
import topognn.data_utils as topodata

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import sys

import argparse

GPU_AVAILABLE = torch.cuda.is_available() and torch.cuda.device_count() > 0


def main(args):

    model_type = args.type

    wandb_logger = WandbLogger(
        name=f"TopoGNN_{args.dataset}", project="topo_gnn", entity="topo_gnn", log_model=True, tags=[args.dataset])

    early_stopping_cb = EarlyStopping(monitor="val_acc", patience=200)
    checkpoint_cb = ModelCheckpoint(
        dirpath=wandb_logger.experiment.dir,
        monitor='val_acc',
        mode='max',
        verbose=True
    )

    trainer = pl.Trainer(
        gpus=-1 if GPU_AVAILABLE else None,
        logger=wandb_logger,
        log_every_n_steps=5,
        max_epochs=args.max_epochs,
        callbacks=[early_stopping_cb, checkpoint_cb]
    )

    if args.dataset == "IMDB-BINARY":
        add_node_degree = True
    else:
        add_node_degree = False

    data = topodata.SyntheticDataset(
        args.dataset, batch_size=32, seed=args.seed, add_node_degree=add_node_degree)
    data.prepare_data()

    num_coord_funs = {"Triangle_transform": args.num_coord_funs,
                      "Gaussian_transform": args.num_coord_funs,
                      "Line_transform": args.num_coord_funs,
                      "RationalHat_transform": args.num_coord_funs
                      }

    num_coord_funs1 = {"Triangle_transform": args.num_coord_funs1,
                       "Gaussian_transform": args.num_coord_funs1,
                       "Line_transform": args.num_coord_funs1,
                       "RationalHat_transform": args.num_coord_funs1
                       }

    model = models.FiltrationGCNModel(hidden_dim=args.hidden_dim,
                                      filtration_hidden=args.filtration_hidden,
                                      num_node_features=data.node_attributes,
                                      num_classes=data.num_classes,
                                      num_filtrations=args.num_filtrations,
                                      num_coord_funs=num_coord_funs,
                                      dim1=args.dim1,
                                      num_coord_funs1=num_coord_funs1,
                                      lr=args.lr,
                                      dropout_p=args.dropout_p,set2set = args.set2set)

    trainer.fit(model, datamodule=data)
    checkpoint_path = checkpoint_cb.best_model_path
    trainer2 = pl.Trainer(logger=False)

    model = models.FiltrationGCNModel.load_from_checkpoint(
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
    parser.add_argument("--type", type=str, default="TopoGNN")
    parser.add_argument("--dim1", type=bool, default=False)
    parser.add_argument("--filtration_hidden", type=int, default=15)
    parser.add_argument("--num_filtrations", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=34)
    parser.add_argument("--num_coord_funs", type=int, default=3)
    parser.add_argument("--num_coord_funs1", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--set2set",type=bool, default = False)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--dataset", type=str, default="Cycles")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.dim1:

        print(f"Using dim1 in the persistence !")
    else:
        print(f"Using dim0 only !")

    if args.set2set:
        print("Using set2set coordinate function")


    main(args)
