import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from dgl.data import LegacyTUDataset


class PTG_LegacyTUDataset(InMemoryDataset):
    @classmethod
    def pretransform_to_ptg(cls, instance):
        g, label = instance
        d = Data(
            x=g.ndata['feat'].float(),
            edge_index=torch.stack(g.edges(), 0),
            y=torch.tensor(label, dtype=torch.long).unsqueeze(0)
        )
        return d

    def __init__(self, root, name, transform=None):
        self.name = name
        super().__init__(root, transform, self.pretransform_to_ptg)
        self.data, self.slices = torch.load(self.processed_paths[0])
        with open(self.processed_paths[1], 'r') as f:
            self._num_classes = int(f.readline()[0])

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt', 'n_classes.txt']

    def download(self):
        LegacyTUDataset(self.name, hidden_size=1)

    def process(self):
        # Read data into huge `Data` list.
        dataset = LegacyTUDataset(self.name, hidden_size=1)
        labels = [instance[1] for instance in dataset]
        n_classes = len(np.unique(labels))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in dataset]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        with open(self.processed_paths[1], 'w') as f:
            f.write(str(n_classes))


if __name__ == '__main__':
    PTG_LegacyTUDataset('test', 'ENZYMES')
