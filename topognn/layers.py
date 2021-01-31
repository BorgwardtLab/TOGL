"""Implementation of layers following Benchmarking GNNs paper."""
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv
from torch_scatter import scatter
from torch_persistent_homology.persistent_homology_cpu import (
    compute_persistence_homology_batched_mt,
)


class GCNLayer(nn.Module):
    def __init__(
        self, in_features, out_features, activation, dropout, batch_norm, residual=True
    ):
        super().__init__()
        self.activation = activation
        self.residual = residual
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(out_features) if batch_norm else nn.Identity()
        self.conv = GCNConv(in_features, out_features, add_self_loops=False)

    def forward(self, x, edge_index, **kwargs):
        h = self.conv(x, edge_index)
        h = self.batchnorm(h)
        h = self.activation(h)
        if self.residual:
            h = h + x
        return self.dropout(h)


class GINLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        activation,
        dropout,
        batch_norm,
        mlp_hidden_dim=None,
        residual=True,
        train_eps=False,
        **kwargs
    ):
        super().__init__()

        if mlp_hidden_dim is None:
            mlp_hidden_dim = in_features

        self.activation = activation
        self.residual = residual
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(out_features) if batch_norm else nn.Identity()
        gin_net = nn.Sequential(
            nn.Linear(in_features, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, out_features),
        )
        self.conv = GINConv(gin_net, train_eps=train_eps)

    def forward(self, x, edge_index, **kwargs):
        h = self.conv(x, edge_index)
        h = self.batchnorm(h)
        if self.residual:
            h = h + x
        return self.dropout(h)


class DeepSetLayer(nn.Module):
    """Simple equivariant deep set layer."""

    def __init__(self, in_dim, out_dim, aggregation_fn):
        super().__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)
        assert aggregation_fn in ["mean", "max", "sum"]
        self.aggregation_fn = aggregation_fn

    def forward(self, x, batch):
        # Apply aggregation function over graph
        xm = scatter(x, batch, dim=0, reduce=self.aggregation_fn)
        xm = self.Lambda(xm)
        x = self.Gamma(x)
        x = x - xm[batch, :]
        return x


class SimpleSetTopoLayer(nn.Module):
    def __init__(self, n_features, n_filtrations, mlp_hidden_dim, aggregation_fn):
        super().__init__()
        self.filtrations = nn.Sequential(
            nn.Linear(n_features, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, n_filtrations),
            nn.BatchNorm1d(n_filtrations),
        )
        self.set_fn0 = nn.ModuleList(
            [
                DeepSetLayer(
                    n_features + n_filtrations * 2, mlp_hidden_dim, aggregation_fn
                ),
                nn.ReLU(),
                DeepSetLayer(mlp_hidden_dim, n_features, aggregation_fn),
            ]
        )
        self.bn = nn.BatchNorm1d(n_features)
        # self.set_fn1 = nn.ModuleList([
        #     DeepSetLayer(n_filtrations*2, mlp_hidden_dim),
        #     nn.ReLU(),
        #     DeepSetLayer(mlp_hidden_dim, n_features),
        # ])

    def compute_persistence(self, x, edge_index, vertex_slices, edge_slices):
        """
        Returns the persistence pairs as a list of tensors with shape [X.shape[0],2].
        The lenght of the list is the number of filtrations.
        """
        filtered_v = self.filtrations(x)
        filtered_e, _ = torch.max(
            torch.stack((filtered_v[edge_index[0]], filtered_v[edge_index[1]])), axis=0
        )

        filtered_v = filtered_v.transpose(1, 0).cpu().contiguous()
        filtered_e = filtered_e.transpose(1, 0).cpu().contiguous()
        edge_index = edge_index.cpu().transpose(1, 0).contiguous()

        persistence0_new, persistence1_new = compute_persistence_homology_batched_mt(
            filtered_v, filtered_e, edge_index, vertex_slices, edge_slices
        )
        persistence0 = persistence0_new.to(x.device)
        persistence1 = persistence1_new.to(x.device)

        return persistence0, persistence1

    def forward(self, x, edge_index, batch, vertex_slices, edge_slices):
        pers0, pers1 = self.compute_persistence(
            x, edge_index, vertex_slices, edge_slices
        )

        x0 = torch.cat([x, pers0.permute(1, 0, 2).reshape(pers0.shape[1], -1)], 1)
        for layer in self.set_fn0:
            if isinstance(layer, DeepSetLayer):
                x0 = layer(x0, batch)
            else:
                x0 = layer(x0)

        # Collect valid
        # valid_0 = (pers1 != 0).all(-1)

        return x + self.bn(x0)
