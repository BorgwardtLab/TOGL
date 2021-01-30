import os
import pytorch_lightning as pl
from topognn import DATA_DIR, Tasks
from torch_geometric.data import DataLoader, Batch, Data
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
from torch_geometric.transforms import OneHotDegree
from torch.utils.data import random_split, Subset
import torch
import math
import pickle
import numpy as np
from torch_geometric.data import InMemoryDataset

from topognn.cli_utils import str2bool
import itertools
from sklearn.model_selection import StratifiedKFold, train_test_split
import csv




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
    task = Tasks.GRAPH_CLASSIFICATION

    def __init__(self, name, batch_size, use_node_attributes=True,
                 val_fraction=0.1, test_fraction=0.1, seed=42, num_workers=4, add_node_degree=False, **kwargs):
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

    @classmethod
    def add_dataset_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--use_node_attributes', type=bool, default=True)
        #parser.add_argument('--fold', type=int, default=0)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--batch_size', type=int, default=32)
        #parser.add_argument('--benchmark_idx',type=str2bool,default=True,help = "If True, uses the idx from the graph benchmarking paper.")
        return parser

def get_label_fromTU(dataset):
    labels = []
    for i in range(len(dataset)):
        labels.append(dataset[i].y)
    return labels


class TUGraphDataset(pl.LightningDataModule):
    task = Tasks.GRAPH_CLASSIFICATION

    def __init__(self, name, batch_size, use_node_attributes=True,
                 val_fraction=0.1, test_fraction=0.1, fold=0, seed=42, num_workers=2, n_splits=5, **kwargs):
        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.use_node_attributes = use_node_attributes
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed
        self.num_workers = num_workers

        max_degrees = {"IMDB-BINARY": 540,
                       "COLLAB": 2000, 'PROTEINS': 50, 'ENZYMES': 18}
        self.has_node_attributes = (
            name not in ['IMDB-BINARY'] and use_node_attributes)
        if not self.has_node_attributes:
            self.max_degree = max_degrees[name]
            self.transform = OneHotDegree(self.max_degree)
        else:
            self.transform = None

        self.n_splits = n_splits
        self.fold = fold

        self.benchmark_idx = kwargs["benchmark_idx"]

    def prepare_data(self):

        if self.name=="PROTEINS_full" or self.name == "ENZYMES":
            cleaned = False
        else:
            cleaned = True

        dataset = TUDataset(
            root=os.path.join(DATA_DIR, self.name),
            use_node_attr=self.has_node_attributes,
            cleaned=cleaned,
            name=self.name,
            transform=self.transform
        )
        self.node_attributes = (
            dataset.num_node_features if self.has_node_attributes
            else self.max_degree + 1
        )
        self.num_classes = dataset.num_classes
        
        if self.benchmark_idx:
            all_idx = {}
            for section in ['train', 'val', 'test']:
                with open(os.path.join(DATA_DIR,'Benchmark_idx',self.name+"_"+section+'.index'),'r') as f:
                    reader = csv.reader(f)
                    all_idx[section] = [list(map(int, idx)) for idx in reader]
            train_index = all_idx["train"][self.fold] 
            val_index = all_idx["val"][self.fold]
            test_index = all_idx["test"][self.fold]
        
        else:
            n_instances = len(dataset)

            skf = StratifiedKFold(n_splits=self.n_splits,
                              random_state=self.seed, shuffle=True)

            skf_iterator = skf.split(
            [i for i in range(n_instances)], get_label_fromTU(dataset))

            train_index, test_index = next(
            itertools.islice(skf_iterator, self.fold, None))
            train_index, val_index = train_test_split(
            train_index, random_state=self.seed)
            
            train_index = train_index.tolist()
            val_index = val_index.tolist()
            test_index = test_index.tolist()

        self.train = Subset(dataset, train_index)
        self.val = Subset(dataset, val_index)
        self.test = Subset(dataset, test_index)

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

    @classmethod
    def add_dataset_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--use_node_attributes', type=bool, default=True)
        parser.add_argument('--fold', type=int, default=0)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--benchmark_idx',type=str2bool,default=True,help = "If True, uses the idx from the graph benchmarking paper.")
        return parser


class IMDB_Binary(TUGraphDataset):
    def __init__(self, **kwargs):
        super().__init__(name='IMDB-BINARY', **kwargs)


class Proteins(TUGraphDataset):
    def __init__(self, **kwargs):
        super().__init__(name='PROTEINS', **kwargs)

class Proteins_full(TUGraphDataset):
    def __init__(self, **kwargs):
        super().__init__(name='PROTEINS_full', **kwargs)

class Enzymes(TUGraphDataset):
    def __init__(self, **kwargs):
        super().__init__(name='ENZYMES', **kwargs)


class DD(TUGraphDataset):
    def __init__(self, **kwargs):
        super().__init__(name='DD', **kwargs)


class MUTAG(TUGraphDataset):
    def __init__(self, **kwargs):
        super().__init__(name='MUTAG', **kwargs)

class Cycles(SyntheticDataset):
    def __init__(self,**kwargs):
        super().__init__(name="Cycles", **kwargs)

class Necklaces(SyntheticDataset):
    def __init__(self,**kwargs):
        super().__init__(name="Necklaces", **kwargs)


def add_pos_to_node_features(instance: Data):
    instance.x = torch.cat([instance.x, instance.pos], axis=-1)
    return instance


class GNNBenchmark(pl.LightningDataModule):
    def __init__(self, name, batch_size, num_workers=4, **kwargs):
        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root = os.path.join(DATA_DIR, self.name)
        if name in ['MNIST', 'CIFAR10']:
            self.task = Tasks.GRAPH_CLASSIFICATION
            self.num_classes = 10
            self.transform = add_pos_to_node_features
        elif name == 'PATTERN':
            self.task = Tasks.NODE_CLASSIFICATION
            self.num_classes = 2
            self.transform = None
        elif name == 'CLUSTER':
            self.task = Tasks.NODE_CLASSIFICATION
            self.num_classes = 6
            self.transform = None
        else:
            raise RuntimeError('Unsupported dataset')

    def prepare_data(self):
        # Just download the data
        train = GNNBenchmarkDataset(
            self.root, self.name, split='train', transform=self.transform)
        self.node_attributes = train[0].x.shape[-1]
        GNNBenchmarkDataset(self.root, self.name, split='val')
        GNNBenchmarkDataset(self.root, self.name, split='test')

    def train_dataloader(self):
        return DataLoader(
            GNNBenchmarkDataset(
                self.root, self.name, split='train', transform=self.transform),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            GNNBenchmarkDataset(
                self.root, self.name, split='val', transform=self.transform),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            GNNBenchmarkDataset(
                self.root, self.name, split='test', transform=self.transform),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
        )

    @classmethod
    def add_dataset_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--batch_size', type=int, default=32)
        return parser


class MNIST(GNNBenchmark):
    def __init__(self, **kwargs):
        super().__init__('MNIST', **kwargs)


class CIFAR10(GNNBenchmark):
    def __init__(self, **kwargs):
        super().__init__('CIFAR10', **kwargs)


class PATTERN(GNNBenchmark):
    def __init__(self, **kwargs):
        super().__init__('PATTERN', **kwargs)


class CLUSTER(GNNBenchmark):
    def __init__(self, **kwargs):
        super().__init__('CLUSTER', **kwargs)
