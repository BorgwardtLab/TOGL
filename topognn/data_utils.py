import os
import pytorch_lightning as pl
from topognn import DATA_DIR
from torch_geometric.data import DataLoader, Batch, Data
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree
from torch.utils.data import random_split
import torch
import math
class TUGraphDataset(pl.LightningDataModule):
    def __init__(self, name, batch_size, use_node_attributes=True,
                 val_fraction=0.1, test_fraction=0.1, seed=42, num_workers=4, add_node_degree = False):
        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.use_node_attributes = use_node_attributes
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed
        self.num_workers = num_workers

        max_degrees = {"IMDB-BINARY":540}
        if add_node_degree:
            self.pre_transform = OneHotDegree(max_degrees[name])
        else:
            self.pre_transform = None

    def prepare_data(self):

        dataset = TUDataset(
            root=os.path.join(DATA_DIR, self.name),
            use_node_attr=True,
            cleaned=self.use_node_attributes,
            name=self.name,
            pre_transform = self.pre_transform
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



