#!/usr/bin/env python
"""Train a model using the same routine as used in the GNN Benchmarks dataset."""
import argparse
import sys

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from topognn.train_model import MODEL_MAP, DATASET_MAP


def main(model_cls, dataset_cls, args):
    # Instantiate objects according to parameters
    dataset = dataset_cls(**vars(args))
    dataset.prepare_data()
   
    if args.set2set:
        raise("Aborting set2set runs")

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
    lr_monitor = LearningRateMonitor('epoch')
    early_stopping_cb = EarlyStopping(monitor="val_loss", patience=500)

    checkpoint_cb = ModelCheckpoint(
        dirpath=wandb_logger.experiment.dir,
        monitor='val_loss',
        mode='min',
        verbose=True
    )

    GPU_AVAILABLE = torch.cuda.is_available() and torch.cuda.device_count() > 0
    trainer = pl.Trainer(
        gpus=-1 if GPU_AVAILABLE else None,
        logger=wandb_logger,
        log_every_n_steps=5,
        max_epochs=args.max_epochs,
        callbacks=[early_stopping_cb, checkpoint_cb, lr_monitor]
    )
    trainer.fit(model, datamodule=dataset)
    test_results = trainer.test(test_dataloaders = dataset.test_dataloader())[0]

    """
    checkpoint_path = checkpoint_cb.best_model_path
    trainer2 = pl.Trainer(logger=False)

    model = model_cls.load_from_checkpoint(
        checkpoint_path)
    val_results = trainer2.test(
        model,
        test_dataloaders=dataset.val_dataloader()
    )[0]

    val_results = {
        name.replace('test', 'best_val'): value
        for name, value in val_results.items()
    }

    test_results = trainer2.test(
        model,
        test_dataloaders=dataset.test_dataloader()
    )[0]

    """

    for name, value in { **test_results}.items():
        wandb_logger.experiment.summary[name] = value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model', type=str, choices=MODEL_MAP.keys())
    parser.add_argument('--dataset', type=str, choices=DATASET_MAP.keys())
    parser.add_argument('--max_epochs', type=int, default=1000)
    partial_args, _ = parser.parse_known_args()

    if partial_args.model is None or partial_args.dataset is None:
        parser.print_usage()
        sys.exit(1)
    model_cls = MODEL_MAP[partial_args.model]
    dataset_cls = DATASET_MAP[partial_args.dataset]

    parser = model_cls.add_model_specific_args(parser)
    parser = dataset_cls.add_dataset_specific_args(parser)
    args = parser.parse_args()

    main(model_cls, dataset_cls, args)
