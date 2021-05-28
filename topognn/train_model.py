#!/usr/bin/env python
"""Train a model, with training procedure similar to Kipf et al."""
import argparse
import sys

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything

import topognn.models as models
import topognn.data_utils as topo_data
from topognn.cli_utils import str2bool

MODEL_MAP = {
    'TopoGNN': models.FiltrationGCNModel,
    'GCN': models.GCNModel,
    'LargerGCN': models.LargerGCNModel,
    'LargerTopoGNN': models.LargerTopoGNNModel,
    'SimpleTopoGNN': models.SimpleTopoGNNModel
}

def main(model_cls, dataset_cls, args):
    args.training_seed = seed_everything(args.training_seed)
    # Instantiate objects according to parameters
    dataset = dataset_cls(**vars(args))
    dataset.prepare_data()

    model = model_cls(
        **vars(args),
        num_node_features=dataset.node_attributes,
        num_classes=dataset.num_classes,
        task=dataset.task
    )
    print('Running with hyperparameters:')
    print(model.hparams)

    # Loggers and callbacks
    wandb_logger = WandbLogger(
        name=f"{args.model}_{args.dataset}",
        project="topo_gnn",
        entity="topo_gnn",
        log_model=True,
        tags=[args.model, args.dataset]
    )
    early_stopping_cb = EarlyStopping(
        monitor="val_loss", patience=args.patience)

    GPU_AVAILABLE = torch.cuda.is_available() and torch.cuda.device_count() > 0
    trainer = pl.Trainer(
        gpus=-1 if GPU_AVAILABLE else None,
        logger=wandb_logger,
        log_every_n_steps=5,
        max_epochs=args.max_epochs,
        callbacks=[early_stopping_cb]
    )
    trainer.fit(model, datamodule=dataset)
    trainer.test(test_dataloaders=dataset.test_dataloader())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model', type=str, choices=MODEL_MAP.keys())
    parser.add_argument('--dataset', type=str, choices=topo_data.dataset_map_dict().keys())
    parser.add_argument('--training_seed', type=int, default=None)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument("--paired", type = str2bool, default=False)
    parser.add_argument("--merged", type = str2bool, default=False)

    partial_args, _ = parser.parse_known_args()

    if partial_args.model is None or partial_args.dataset is None:
        parser.print_usage()
        sys.exit(1)
    model_cls = MODEL_MAP[partial_args.model]
    dataset_cls = topo_data.get_dataset_class(**vars(partial_args))

    parser = model_cls.add_model_specific_args(parser)
    parser = dataset_cls.add_dataset_specific_args(parser)
    args = parser.parse_args()

    main(model_cls, dataset_cls, args)
