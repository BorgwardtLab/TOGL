import os
import pytorch_lightning as pl
from topognn import DATA_DIR
from torch_geometric.data import DataLoader, Batch, Data
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree
from torch.utils.data import random_split
import torch
import math
import pickle
from torch_geometric.data import InMemoryDataset


class SyntheticBaseDataset(InMemoryDataset):
    def __init__(self, root = DATA_DIR, transform=None, pre_transform=None):
        super(SyntheticBaseDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['graphs.txt','labels.pt']

    @property
    def processed_file_names(self):
        return ['synthetic_data.pt']

    #def download(self):
    #    # Download to `self.raw_dir`.
    #    raise("Not Implemented")

    def process(self):
        # Read data into huge `Data` list.
        with open(f"{self.root}/graphs.txt", "rb") as fp:   # Unpickling
            x_list, edge_list = pickle.load(fp)
            
        labels = torch.load(f"{self.root}/labels.pt")
        data_list = [Data(x=x_list[i], edge_index=edge_list[i], y = labels[i][None]) for i in range(len(x_list))]
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



class SyntheticDataset(pl.LightningDataModule):
    def __init__(self, name, batch_size, use_node_attributes=True,
                 val_fraction=0.1, test_fraction=0.1, seed=42, num_workers=4, add_node_degree = False):
        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed
        self.num_workers = num_workers

        if add_node_degree:
            self.pre_transform = OneHotDegree(max_degrees[name])
        else:
            self.pre_transform = None

    def prepare_data(self):

        dataset = SyntheticBaseDataset(
            root=os.path.join(DATA_DIR,"SYNTHETIC", self.name),
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



