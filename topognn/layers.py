"""Implementation of layers following Benchmarking GNNs paper."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv
from torch_scatter import scatter
from torch_persistent_homology.persistent_homology_cpu import (
    compute_persistence_homology_batched_mt,
)
from topognn.data_utils import remove_duplicate_edges


class GCNLayer(nn.Module):
    def __init__(
        self, in_features, out_features, activation, dropout, batch_norm, residual=True
    ):
        super().__init__()
        self.activation = activation
        self.residual = residual
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(
            out_features) if batch_norm else nn.Identity()
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
        self.batchnorm = nn.BatchNorm1d(
            out_features) if batch_norm else nn.Identity()
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


class DeepSetLayerDim1(nn.Module):
    """Simple equivariant deep set layer."""

    def __init__(self, in_dim, out_dim, aggregation_fn):
        super().__init__()
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)
        assert aggregation_fn in ["mean", "max", "sum"]
        self.aggregation_fn = aggregation_fn

    def forward(self, x, edge_slices, mask=None):
        '''
        Mask is True where the persistence (x) is observed.
        '''
        # Apply aggregation function over graph

        # Computing the equivalent of batch over edges.
        edge_diff_slices = (edge_slices[1:]-edge_slices[:-1]).to(x.device)
        n_batch = len(edge_diff_slices)
        batch_e = torch.repeat_interleave(torch.arange(
            n_batch, device=x.device), edge_diff_slices)
        # Only aggregate over edges with non zero persistence pairs.
        if mask is not None:
            batch_e = batch_e[mask]

        xm = scatter(x, batch_e, dim=0,
                     reduce=self.aggregation_fn, dim_size=n_batch)

        xm = self.Lambda(xm)

        # xm = scatter(x, batch, dim=0, reduce=self.aggregation_fn)
        # xm = self.Lambda(xm)
        # x = self.Gamma(x)
        # x = x - xm[batch, :]
        return xm


def fake_persistence_computation(filtered_v_, edge_index, vertex_slices, edge_slices, batch):
    device = filtered_v_.device
    num_filtrations = filtered_v_.shape[1]
    filtered_e_, _ = torch.max(torch.stack(
        (filtered_v_[edge_index[0]], filtered_v_[edge_index[1]])), axis=0)

    # Make fake tuples for dim 0
    persistence0_new = filtered_v_.unsqueeze(-1).expand(-1, -1, 2)

    # Make fake dim1 with unpaired values
    unpaired_values = scatter(filtered_v_, batch, dim=0, reduce='max')
    persistence1_new = torch.zeros(
        edge_index.shape[1], filtered_v_.shape[1], 2, device=device)

    edge_slices = edge_slices.to(device)
    bs = edge_slices.shape[0] - 1
    n_edges = edge_slices[1:] - edge_slices[:-1]
    random_edges = (
        edge_slices[0:-1].unsqueeze(-1) +
        torch.floor(
            torch.rand(size=(bs, num_filtrations), device=device)
            * n_edges.float().unsqueeze(-1)
        )
    ).long()

    persistence1_new[random_edges, torch.arange(num_filtrations).unsqueeze(0), :] = (
        torch.stack([
            unpaired_values,
            filtered_e_[
                    random_edges, torch.arange(num_filtrations).unsqueeze(0)]
        ], -1)
    )
    return persistence0_new.permute(1, 0, 2), persistence1_new.permute(1, 0, 2)


class SimpleSetTopoLayer(nn.Module):
    def __init__(self, n_features, n_filtrations, mlp_hidden_dim,
                 aggregation_fn, dim0_out_dim, dim1_out_dim, dim1,
                 residual_and_bn, fake, deepset_type='full',
                 swap_bn_order=False, dist_dim1=False):
        super().__init__()
        self.filtrations = nn.Sequential(
            nn.Linear(n_features, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, n_filtrations),
        )

        assert deepset_type in ['linear', 'shallow', 'full']

        self.num_filtrations = n_filtrations
        self.residual_and_bn = residual_and_bn
        self.swap_bn_order = swap_bn_order
        self.dist_dim1 = dist_dim1

        self.dim1_flag = dim1
        if self.dim1_flag:
            if deepset_type == 'linear':
                self.set_fn1 = nn.ModuleList([nn.Linear(
                    n_filtrations * 2,
                    n_features if residual_and_bn and dist_dim1 else dim1_out_dim
                )])
            else:
                self.set_fn1 = nn.ModuleList([
                    nn.Linear(n_filtrations * 2, dim1_out_dim),
                    nn.ReLU(),
                    DeepSetLayerDim1(
                        in_dim=dim1_out_dim, out_dim=n_features if residual_and_bn and dist_dim1 else dim1_out_dim, aggregation_fn=aggregation_fn),
                    nn.ReLU()
                ])

        if deepset_type == 'linear':
            self.set_fn0 = nn.ModuleList([nn.Linear(
                n_filtrations * 2,
                n_features if residual_and_bn else dim0_out_dim, aggregation_fn)
            ])
        elif deepset_type == 'shallow':
            self.set_fn0 = nn.ModuleList(
                [
                    nn.Linear(n_filtrations * 2, dim0_out_dim),
                    nn.ReLU(),
                    DeepSetLayer(
                        dim0_out_dim, n_features if residual_and_bn else dim0_out_dim, aggregation_fn),
                ]
            )
        else:
            self.set_fn0 = nn.ModuleList(
                [
                    nn.Linear(n_filtrations * 2, dim0_out_dim),
                    nn.ReLU(),
                    DeepSetLayer(dim0_out_dim, dim0_out_dim, aggregation_fn),
                    nn.ReLU(),
                    DeepSetLayer(
                        dim0_out_dim, n_features if residual_and_bn else dim0_out_dim, aggregation_fn),
                ]
            )
        if residual_and_bn:
            self.bn = nn.BatchNorm1d(n_features)
        else:
            if dist_dim1:
                self.out = nn.Sequential(
                    nn.Linear(dim0_out_dim + dim1_out_dim +
                              n_features, n_features),
                    nn.ReLU()
                )
            else:
                self.out = nn.Sequential(
                    nn.Linear(dim0_out_dim + n_features, n_features),
                    nn.ReLU()
                )
        self.fake = fake

    def compute_persistence(self, x, edge_index, vertex_slices, edge_slices, batch):
        """
        Returns the persistence pairs as a list of tensors with shape [X.shape[0],2].
        The lenght of the list is the number of filtrations.
        """
        filtered_v = self.filtrations(x)
        if self.fake:
            return fake_persistence_computation(
                filtered_v, edge_index, vertex_slices, edge_slices, batch)

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

    def forward(self, x, data):

        # Remove the duplucate edges
        data = remove_duplicate_edges(data)

        edge_index = data.edge_index
        vertex_slices = torch.Tensor(data.__slices__['x']).cpu().long()
        edge_slices = torch.Tensor(data.__slices__['edge_index']).cpu().long()
        batch = data.batch

        pers0, pers1 = self.compute_persistence(
            x, edge_index, vertex_slices, edge_slices, batch
        )

        x0 = pers0.permute(1, 0, 2).reshape(pers0.shape[1], -1)

        for layer in self.set_fn0:
            if isinstance(layer, DeepSetLayer):
                x0 = layer(x0, batch)
            else:
                x0 = layer(x0)

        if self.dim1_flag:
            # Dim 1 computations.
            pers1_reshaped = pers1.permute(1, 0, 2).reshape(pers1.shape[1], -1)
            pers1_mask = ~((pers1_reshaped == 0).all(-1))
            x1 = pers1_reshaped[pers1_mask]
            for layer in self.set_fn1:
                if isinstance(layer, DeepSetLayerDim1):
                    x1 = layer(x1, edge_slices, mask=pers1_mask)
                else:
                    x1 = layer(x1)
        else:
            x1 = None

        # Collect valid
        # valid_0 = (pers1 != 0).all(-1)

        if self.residual_and_bn:
            if self.dist_dim1 and self.dim1_flag:
                x0 = x0 + x1[batch]
                x1 = None
            if self.swap_bn_order:
                x = x + F.relu(self.bn(x0))
            else:
                x = x + self.bn(F.relu(x0))
        else:
            if self.dist_dim1 and self.dim1_flag:
                x0 = torch.cat([x0, x1[batch]], dim=-1)
                x1 = None
            x = self.out(torch.cat([x, x0], dim=-1))

        return x, x1
