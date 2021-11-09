"""Utility functions for data sets."""

import csv
import itertools
import math
import os
import pickle
import torch

import networkx as nx
import numpy as np
import pytorch_lightning as pl

from topognn import DATA_DIR, Tasks
from topognn.cli_utils import str2bool

from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset, Planetoid
from torch_geometric.transforms import OneHotDegree
from torch_geometric.utils import degree
from torch_geometric.utils.convert import from_networkx
from torch_geometric import transforms

from torch_scatter import scatter

from torch.utils.data import random_split, Subset

from sklearn.model_selection import StratifiedKFold, train_test_split

from ogb.graphproppred import PygGraphPropPredDataset


def dataset_map_dict():
    DATASET_MAP = {
        'IMDB-BINARY': IMDB_Binary,
        'IMDB-MULTI': IMDB_Multi,
        'REDDIT-BINARY': REDDIT_Binary,
        'REDDIT-5K': REDDIT_5K,
        'PROTEINS': Proteins,
        'PROTEINS_full': Proteins_full,
        'ENZYMES': Enzymes,
        'DD': DD,
        'NCI1' : NCI,
        'MUTAG': MUTAG,
        'MNIST': MNIST,
        'CIFAR10': CIFAR10,
        'PATTERN': PATTERN,
        'CLUSTER': CLUSTER,
        'Necklaces': Necklaces,
        'Cycles': Cycles,
        'NoCycles': NoCycles,
        'CliquePlanting': CliquePlanting,
        'DBLP': DBLP,
        'Cora': Cora,
        'CiteSeer' : CiteSeer,
        'PubMed': PubMed,
        'MOLHIV': MOLHIV
    }

    return DATASET_MAP


def remove_duplicate_edges(batch):

        with torch.no_grad():
            batch = batch.clone()        
            device = batch.x.device
            # Computing the equivalent of batch over edges.
            edge_slices = torch.tensor(batch.__slices__["edge_index"],device= device)
            edge_diff_slices = (edge_slices[1:]-edge_slices[:-1])
            n_batch = len(edge_diff_slices)
            batch_e = torch.repeat_interleave(torch.arange(
                n_batch, device = device), edge_diff_slices)

            correct_idx = batch.edge_index[0] <= batch.edge_index[1]
            #batch_e_idx = batch_e[correct_idx]
            n_edges = scatter(correct_idx.long(), batch_e, reduce = "sum")
           
            batch.edge_index = batch.edge_index[:,correct_idx]
           
            new_slices = torch.cumsum(torch.cat((torch.zeros(1,device=device, dtype=torch.long),n_edges)),0).tolist()

            batch.__slices__["edge_index"] =  new_slices     
            return batch

def get_dataset_class(**kwargs):

    if kwargs.get("paired", False):
        # (kwargs["dataset"],batch_size = kwargs["batch_size"], disjoint = not kwargs["merged"] )
        dataset_cls = PairedTUGraphDataset
    else:

        dataset_cls = dataset_map_dict()[kwargs["dataset"]]
    return dataset_cls


class CliquePlantingDataset(InMemoryDataset):
    """Clique planting data set."""

    def __init__(
        self,
        root,
        n_graphs=1000,
        n_vertices=100,
        k_clique=17,
        random_d = 3,
        p_ER_graph = 0.5,
        pre_transform=None,
        transform=None,
        **kwargs
    ):
        """Initialise new variant of clique planting data set.

        Parameters
        ----------
        root : str
            Root directory for storing graphs.

        n_graphs : int
            How many graphs to create.

        n_vertices : int
            Size of graph for planting a clique.

        k : int
            Size of clique. Must be subtly 'compatible' with n, but the
            class will warn if problematic values are being chosen.
        """
        self.n_graphs = n_graphs
        self.n_vertices = n_vertices
        self.k = k_clique
        self.random_d = random_d
        self.p = p_ER_graph

        super().__init__(root)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """No raw file names are required."""
        return []

    @property
    def processed_dir(self):
        """Directory to store data in."""
        # Following the other classes, we are relying on the client to
        # provide a proper path.
        return os.path.join(
            self.root,
            'processed'
        )

    @property
    def processed_file_names(self):
        """Return file names for identification of stored data."""
        N = self.n_graphs
        n = self.n_vertices
        k = self.k
        return [f'data_{N}_{n}_{k}.pt']

    def process(self):
        """Create data set and store it in memory for subsequent processing."""
        graphs = [self._make_graph() for i in range(self.n_graphs)]
        labels = [y for _, y in graphs]

        data_list = [from_networkx(g) for g, _ in graphs]
        for data, label in zip(data_list, labels):
            data.y = label
            data.x = torch.randn(data.num_nodes,self.random_d)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _make_graph(self):
        """Create graph potentially containing a planted clique."""
        G = nx.erdos_renyi_graph(self.n_vertices, p=self.p)
        y = 0
        #nx.classes.function.set_node_attributes(G,dict(G.degree),name="degree")
        
        if np.random.choice([True, False]):
            G = self._plant_clique(G, self.k)
            y = 1

        return G, y

    def _plant_clique(self, G, k):
        """Plant $k$-clique in a given graph G.

        This function chooses a random subset of the vertices of the graph and
        turns them into fully-connected subgraph.
        """
        n = G.number_of_nodes()
        vertices = np.random.choice(np.arange(n), k, replace=False)

        for index, u in enumerate(vertices):
            for v in vertices[index+1:]:
                G.add_edge(u, v)

        return G


class SyntheticBaseDataset(InMemoryDataset):
    def __init__(self, root=DATA_DIR, transform=None, pre_transform=None, **kwargs):
        super(SyntheticBaseDataset, self).__init__(
            root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['graphs.txt', 'labels.pt']

    @property
    def processed_file_names(self):
        return ['synthetic_data.pt']

    def process(self):
        # Read data into huge `Data` list.
        with open(f"{self.root}/graphs.txt", "rb") as fp:   # Unpickling
            x_list, edge_list = pickle.load(fp)

        labels = torch.load(f"{self.root}/labels.pt")
        data_list = [Data(x=x_list[i], edge_index=edge_list[i],
                          y=labels[i][None]) for i in range(len(x_list))]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SyntheticDataset(pl.LightningDataModule):
    task = Tasks.GRAPH_CLASSIFICATION

    def __init__(
        self,
        name,
        batch_size,
        use_node_attributes=True,
        val_fraction=0.1,
        test_fraction=0.1,
        seed=42,
        num_workers=4,
        add_node_degree=False,
        dataset_class=SyntheticBaseDataset,
        **kwargs
    ):
        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed
        self.num_workers = num_workers
        self.dataset_class = dataset_class
        self.kwargs = kwargs

        if add_node_degree:
            self.pre_transform = OneHotDegree(max_degrees[name])
        else:
            self.pre_transform = None

    def prepare_data(self):
        """Load or create data set according to the provided parameters."""
        dataset = self.dataset_class(
            root=os.path.join(DATA_DIR, 'SYNTHETIC', self.name),
            pre_transform=self.pre_transform,
            **self.kwargs
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
        parser.add_argument('--min_cycle',type=int,default = 3)
        parser.add_argument('--k_clique',type=int, default = 17)
        parser.add_argument('--p_ER_graph',type=float, default = 0.5, help = "Probability of an edge in the ER graph (only for CliquePlanting)")
        #parser.add_argument('--benchmark_idx',type=str2bool,default=True,help = "If True, uses the idx from the graph benchmarking paper.")
        return parser


def get_label_fromTU(dataset):
    labels = []
    for i in range(len(dataset)):
        labels.append(dataset[i].y)
    return labels


def get_degrees_fromTU(name):

    dataset = TUDataset(
            root=os.path.join(DATA_DIR, name),
            use_node_attr=True,
            cleaned=True,
            name = name)
    degs = []
    for data in dataset:
        degs += [degree(data.edge_index[0], dtype=torch.long)]
    

    deg = torch.cat(degs, dim=0).to(torch.float)
    mean, std = deg.mean().item(), deg.std().item()

    print(f"Mean of degree of {name} = {mean} with std : {std}")

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

class RandomAttributes(object):
    def __init__(self,d):
        self.d = d
    def __call__(self,data):
        data.x = torch.randn((data.x.shape[0],self.d))
        return data

class OGBDataset(pl.LightningDataModule):
    def __init__(self, name, batch_size, use_node_attributes=True,
                 fold=0, seed=42,
                 num_workers=2, **kwargs):
        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.use_node_attributes = use_node_attributes
        self.fold = fold
        self.num_workers = num_workers

    def prepare_data(self):

        if not self.use_node_attributes:
            self.transform = RandomAttributes(d=3)
            self.node_attributes = 3
        else:
            self.transform = None

        dataset = PygGraphPropPredDataset(name = self.name, root = os.path.join(DATA_DIR, self.name), transform = self.transform)
        if self.use_node_attributes:
            self.node_attributes = dataset.data.x.shape[1]
        
        self.num_classes = int(dataset.meta_info["num classes"])
        self.task = Tasks.GRAPH_CLASSIFICATION
        split_idx = dataset.get_idx_split()
        self.train = dataset[split_idx["train"]]
        self.val = dataset[split_idx["valid"]]
        self.test = dataset[split_idx["test"]]

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
        parser.add_argument('--use_node_attributes', type=str2bool, default=True)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--batch_size', type=int, default=32)
        return parser




class TUGraphDataset(pl.LightningDataModule):
    #task = Tasks.GRAPH_CLASSIFICATION

    def __init__(self, name, batch_size, use_node_attributes=True,
                 val_fraction=0.1, test_fraction=0.1, fold=0, seed=42,
                 num_workers=2, n_splits=5, legacy=True, **kwargs):

        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.use_node_attributes = use_node_attributes
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed
        self.num_workers = num_workers
        self.legacy = legacy

        if name == "DBLP_v1":
            self.task = Tasks.NODE_CLASSIFICATION_WEIGHTED
        else:
            self.task = Tasks.GRAPH_CLASSIFICATION

        max_degrees = {"IMDB-BINARY": 540,
                "COLLAB": 2000, 'PROTEINS': 50, 'ENZYMES': 18, "REDDIT-BINARY": 12200, "REDDIT-MULTI-5K":8000, "IMDB-MULTI":352}
        mean_degrees = {"REDDIT-BINARY":2.31,"REDDIT-MULTI-5K":2.34}
        std_degrees  = {"REDDIT-BINARY": 20.66,"REDDIT-MULTI-5K":12.50}

        self.has_node_attributes = use_node_attributes

        self.pre_transform = None

        if not use_node_attributes:
            self.transform = RandomAttributes(d=3)
            #self.transform = None
        else:
            if name in ['IMDB-BINARY','IMDB-MULTI','REDDIT-BINARY','REDDIT-MULTI-5K']:
                self.max_degree = max_degrees[name]
                if self.max_degree < 1000:
                    self.pre_transform = OneHotDegree(self.max_degree)
                    self.transform = None
                else:
                    self.transform = None
                    self.pre_transform = NormalizedDegree(mean_degrees[name],std_degrees[name])

            else:
                self.transform = None

        self.n_splits = n_splits
        self.fold = fold

        if name in ["PROTEINS_full", "ENZYMES", "DD"]:
            self.benchmark_idx = kwargs["benchmark_idx"]
        else:
            self.benchmark_idx = False

    def prepare_data(self):
        from topognn.tu_datasets import PTG_LegacyTUDataset

        if self.name == "PROTEINS_full" or self.name == "ENZYMES":
            cleaned = False
        else:
            cleaned = True


        if self.legacy:
            dataset = PTG_LegacyTUDataset(
                root=os.path.join(DATA_DIR, self.name + '_legacy'),
                # use_node_attr=self.has_node_attributes,
                # cleaned=cleaned,
                name=self.name,
                transform=self.transform
            )
            self.node_attributes = dataset[0].x.shape[1]
        else:
            dataset = TUDataset(
                root=os.path.join(DATA_DIR, self.name),
                use_node_attr=self.has_node_attributes,
                cleaned=cleaned,
                name=self.name,
                transform=self.transform,
                pre_transform = self.pre_transform
            )
            
            if self.has_node_attributes:
                self.node_attributes= dataset.num_node_features
            else:
                if self.max_degree<1000:
                    self.node_attributes = self.max_degree+1
                else:
                    self.node_attributes = dataset.num_node_features

        self.num_classes = dataset.num_classes

        if self.benchmark_idx:
            all_idx = {}
            for section in ['train', 'val', 'test']:
                with open(os.path.join(DATA_DIR, 'Benchmark_idx', self.name+"_"+section+'.index'), 'r') as f:
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
                torch.tensor([i for i in range(n_instances)]), torch.tensor(get_label_fromTU(dataset)))


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
        parser.add_argument('--use_node_attributes', type=str2bool, default=True)
        parser.add_argument('--fold', type=int, default=0)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--legacy', type=str2bool, default=True)
        parser.add_argument('--benchmark_idx', type=str2bool, default=True,
                            help="If True, uses the idx from the graph benchmarking paper.")
        return parser


class PairedTUGraphDatasetBase(TUDataset):
    """Pair graphs in TU data set."""

    def __init__(self, name, disjoint, **kwargs):
        """Create new paired graph data set from named TU data set.

        Parameters
        ----------
        name : str
            Name of the TU data set to use as the parent data set. Must
            be a data set with a binary classification task.

        disjoint : bool
            If set, performs a disjoint union between the two graphs
            that are supposed to be paired, resulting in two connected
            components.

        **kwargs : kwargs
            Optional set of keyword arguments that will be used for
            loading the parent TU data set.
        """
        # We only require this for the pairing routine; by default, the
        # disjoint union of graphs will be calculated.
        self.disjoint = disjoint

        if name == "PROTEINS_full" or name == "ENZYMES":
            cleaned = False
        else:
            cleaned = True

        root = os.path.join(DATA_DIR, name)

        super().__init__(name=name, root=root, cleaned=cleaned, **kwargs)

    def _pair_graphs(self):
        """Auxiliary function for performing graph pairing.

        Returns
        -------
        Tuple of data tensor and slices array, which can be saved to the
        disk or used for further processing.
        """
        y = self.data.y.numpy()

        # Some sanity checks before continuing with processing the data
        # set.
        labels = sorted(np.unique(self.data.y))
        n_classes = len(labels)
        if n_classes != 2:
            raise RuntimeError(
                'Paired data set is only defined for binary graph '
                'classification tasks.'
            )

        # Will contain the merged graphs as single `Data` objects,
        # consisting of proper pairings of the respective inputs.
        data = []

        for i, label in enumerate(y):
            partners = np.arange(len(y))
            partners = partners[i < partners]

            for j in partners:

                # FIXME
                #
                # Cannot use `int64` to access the data set. I am
                # reasonably sure that this is *wrong*.
                j = int(j)

                # Merge the two graphs into a single graph with two
                # connected components. This requires merges of all
                # the tensors (except for `y`, which we *know*, and
                # `edge_index`, which we have to merge in dimension
                # 1 instead of 0).

                merged = {}

                # Offset all nodes of the second graph correctly to
                # ensure that we will get new edges and no isolated
                # nodes.
                offset = self[i].num_nodes
                edge_index = torch.cat(
                    (self[i].edge_index, self[j].edge_index + offset),
                    1
                )

                new_label = int(label == y[j])

                # Only graphs whose components stem from the positive
                # class will be accepted here; put *all* other graphs
                # into the negative class.
                if label != 1:
                    new_label = 0

                # Check whether we are dealing with the positive label,
                # i.e. the last of the unique labels, when creating the
                # set of *merged* graphs.
                if not self.disjoint and new_label == 1:
                    u = torch.randint(0, self[i].num_nodes, (1,))
                    v = torch.randint(0, self[j].num_nodes, (1,)) + offset

                    edge = torch.tensor([[u], [v]], dtype=torch.long)
                    edge_index = torch.cat((edge_index, edge), 1)

                merged['edge_index'] = edge_index
                merged['y'] = torch.tensor([new_label], dtype=torch.long)

                for attr_name in dir(self[i]):

                    # No need to merge labels or edge_indices
                    if attr_name == 'y' or attr_name == 'edge_index':
                        continue

                    attr = getattr(self[i], attr_name)

                    if type(attr) == torch.Tensor:
                        merged[attr_name] = torch.cat(
                            (
                                getattr(self[i], attr_name),
                                getattr(self[j], attr_name)
                            ), 0
                        )

                data.append(Data(**merged))

        data, slices = self.collate(data)
        return data, slices

    def download(self):
        """Download data set."""
        super().download()

    @property
    def processed_dir(self):
        """Return name of directory for storing paired graphs."""
        name = 'paired{}{}'.format(
            '_cleaned' if self.cleaned else '',
            '_merged' if not self.disjoint else ''
        )
        return os.path.join(self.root, self.name, name)

    def process(self):
        """Process data set according to input parameters."""
        # First finish everything in the parent data set before starting
        # to pair the graphs and write them out.
        super().process()

        self.data, self.slices = self._pair_graphs()
        torch.save((self.data, self.slices), self.processed_paths[0])


class PairedTUGraphDataset(pl.LightningDataModule):
    task = Tasks.GRAPH_CLASSIFICATION

    def __init__(
        self,
        dataset,
        batch_size,
        use_node_attributes=True,
        merged=False,
        val_fraction=0.1,
        test_fraction=0.1,
        seed=42,
        num_workers=4,
        **kwargs
    ):
        """Create new paired data set."""
        super().__init__()

        self.name = dataset
        self.disjoint = not merged
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed
        self.num_workers = num_workers
        self.use_node_attributes = use_node_attributes

    def prepare_data(self):
        dataset = PairedTUGraphDatasetBase(
            self.name,
            disjoint=self.disjoint,
            use_node_attr=self.use_node_attributes,
        )

        self.node_attributes = dataset.num_node_features
        self.num_classes = dataset.num_classes
        n_instances = len(dataset)

        # FIXME: should this be updated?
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
        parser.add_argument('--use_node_attributes', type=str2bool, default=True)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--batch_size', type=int, default=32)
        return parser


class IMDB_Binary(TUGraphDataset):
    def __init__(self, **kwargs):
        super().__init__(name='IMDB-BINARY', **kwargs)

class IMDB_Multi(TUGraphDataset):
    def __init__(self, **kwargs):
        super().__init__(name='IMDB-MULTI', **kwargs)

class REDDIT_Binary(TUGraphDataset):
    def __init__(self, **kwargs):
        super().__init__(name='REDDIT-BINARY', **kwargs)

class REDDIT_5K(TUGraphDataset):
    def __init__(self, **kwargs):
        super().__init__(name='REDDIT-MULTI-5K', **kwargs)

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

class NCI(TUGraphDataset):
    def __init__(self, **kwargs):
        super().__init__(name='NCI1', **kwargs)

class DBLP(TUGraphDataset):
    def __init__(self, **kwargs):
        super().__init__(name='DBLP_v1', **kwargs)

class MOLHIV(OGBDataset):
    def __init__(self, **kwargs):
        super().__init__(name='ogbg-molhiv', **kwargs)

class Cycles(SyntheticDataset):
    def __init__(self, min_cycle, **kwargs):
        name = "Cycles" + f"_{min_cycle}"
        super().__init__(name=name, **kwargs)

class NoCycles(SyntheticDataset):
    def __init__(self, **kwargs):
        super().__init__(name="NoCycles", **kwargs)

class Necklaces(SyntheticDataset):
    def __init__(self, **kwargs):
        super().__init__(name="Necklaces", **kwargs)

class CliquePlanting(SyntheticDataset):
    def __init__(self, **kwargs):
        
        super().__init__(
            name="CliquePlanting",
            dataset_class=CliquePlantingDataset,
            **kwargs
        )

def add_pos_to_node_features(instance: Data):
    instance.x = torch.cat([instance.x, instance.pos], axis=-1)
    return instance


class GNNBenchmark(pl.LightningDataModule):
    def __init__(self, name, batch_size, use_node_attributes, num_workers=4, **kwargs):
        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root = os.path.join(DATA_DIR, self.name)

        self.transforms_list = []
        
        if name in ['MNIST', 'CIFAR10']:
            self.task = Tasks.GRAPH_CLASSIFICATION
            self.num_classes = 10
            if use_node_attributes:
                self.transforms_list.append(add_pos_to_node_features)
        elif name == 'PATTERN':
            self.task = Tasks.NODE_CLASSIFICATION_WEIGHTED
            self.num_classes = 2
            if use_node_attributes is False:
                self.transforms_list.append(RandomAttributes(d=3))
        elif name == 'CLUSTER':
            self.task = Tasks.NODE_CLASSIFICATION_WEIGHTED
            self.num_classes = 6
            if use_node_attributes is False:
                self.transforms_list.append(RandomAttributes(d=3))
        else:
            raise RuntimeError('Unsupported dataset')

        if len(self.transforms_list)>0:
            self.transform  = transforms.Compose(self.transforms_list)
        else:
            self.transform = None

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
        parser.add_argument('--use_node_attributes', type=str2bool, default=True)
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

class PlanetoidDataset(pl.LightningDataModule):
    def __init__(self, name, use_node_attributes, num_workers=4, **kwargs):
        super().__init__()
        self.name = name
        self.num_workers = num_workers
        self.root = os.path.join(DATA_DIR, self.name)

        self.task = Tasks.NODE_CLASSIFICATION

        if use_node_attributes:
            self.random_transform = lambda x : x
        else:
            self.random_transform = RandomAttributes(d=3)

    def prepare_data(self):
        # Just download the data
        dummy_data = Planetoid(
                self.root, self.name, split='public', transform=transforms.Compose([self.random_transform, PlanetoidDataset.keep_train_transform]))
        self.num_classes = int(torch.max(dummy_data[0].y) + 1)
        self.node_attributes = dummy_data[0].x.shape[1]
        return

    def train_dataloader(self):
        return DataLoader(
            Planetoid(
                    self.root,
                    self.name,
                    split='public',
                    transform=transforms.Compose([self.random_transform, PlanetoidDataset.keep_train_transform])
            ),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            Planetoid(
                self.root,
                self.name,
                split='public',
                transform=transforms.Compose([self.random_transform, PlanetoidDataset.keep_val_transform])
            ),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            Planetoid(
                self.root,
                self.name,
                split='public',
                transform=transforms.Compose([self.random_transform, PlanetoidDataset.keep_test_transform])
            ),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=True
        )

    @staticmethod
    def keep_train_transform(data):
        data.y[~data.train_mask] = -100
        return data

    def keep_val_transform(data):
        data.y[~data.val_mask] = -100
        return data

    def keep_test_transform(data):
        data.y[~data.test_mask] = -100
        return data

    @classmethod
    def add_dataset_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--use_node_attributes', type=str2bool, default=True)
        return parser


class Cora(PlanetoidDataset):
    def __init__(self, **kwargs):
        super().__init__(name='Cora', split = "public", **kwargs)

class CiteSeer(PlanetoidDataset):
    def __init__(self, **kwargs):
        super().__init__(name='CiteSeer', split = "public", **kwargs)

class PubMed(PlanetoidDataset):
    def __init__(self, **kwargs):
        super().__init__(name='PubMed', split = "public", **kwargs)
