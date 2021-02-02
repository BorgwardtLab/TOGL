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

    def __init__(self, in_dim, out_dim, aggregation_fn, **kwargs):
        super().__init__()
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)
        assert aggregation_fn in ["mean", "max", "sum"]
        self.aggregation_fn = aggregation_fn

    def forward(self, x, edge_slices, mask = None):
        '''
        Mask is True where the persistence (x) is observed.
        '''
        # Apply aggregation function over graph
        
        #Computing the equivalent of batch over edges.
        edge_diff_slices = (edge_slices[1:]-edge_slices[:-1]).to(x.device)
        n_batch = len(edge_diff_slices)
        batch_e = torch.repeat_interleave(torch.arange(n_batch, device = x.device),edge_diff_slices)
        #Only aggregate over edges with non zero persistence pairs.
        if mask is not None:
            batch_e = batch_e[mask]

        xm = scatter(x, batch_e, dim= 0, reduce=self.aggregation_fn, dim_size= n_batch)
        
        xm = self.Lambda(xm)

        #xm = scatter(x, batch, dim=0, reduce=self.aggregation_fn)
        #xm = self.Lambda(xm)
        #x = self.Gamma(x)
        #x = x - xm[batch, :]
        return xm




class SimpleSetTopoLayer(nn.Module):
    def __init__(self, n_features, n_filtrations, mlp_hidden_dim, aggregation_fn, dim0_out_dim, dim1_out_dim, dim1, fake, **kwargs):
        super().__init__()
        self.filtrations = nn.Sequential(
            nn.Linear(n_features, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, n_filtrations),
        )

        self.num_filtrations = n_filtrations

        self.dim1_flag = dim1
        if self.dim1_flag:
            self.dim1_fn = DeepSetLayerDim1(in_dim =  2 * n_features + n_filtrations*2,
                out_dim = dim1_out_dim,
                aggregation_fn = aggregation_fn,
                **kwargs)

        self.set_fn0 = nn.ModuleList(
            [
                DeepSetLayer(
                    n_features + n_filtrations * 2, dim0_out_dim, aggregation_fn
                ),
                nn.ReLU(),
                DeepSetLayer(dim0_out_dim, n_features, aggregation_fn),
            ]
        )
        self.bn = nn.BatchNorm1d(n_features)

        self.fake = fake
        # self.set_fn1 = nn.ModuleList([
        #     DeepSetLayer(n_filtrations*2, mlp_hidden_dim),
        #     nn.ReLU(),
        #     DeepSetLayer(mlp_hidden_dim, n_features),
        # ])

    def compute_persistence(self, x, edge_index, vertex_slices, edge_slices, batch):
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

    def compute_fake_persistence(self,x,edge_index, vertex_slices, edge_slices, batch):
        
        filtered_v_ = self.filtrations(x)
        
        filtered_e_, _ = torch.max(torch.stack(
            (filtered_v_[edge_index[0]], filtered_v_[edge_index[1]])), axis=0)

        # Make fake tuples for dim 0
        persistence0_new = filtered_v_.unsqueeze(-1).expand(-1, -1, 2)
        
        # Make fake dim1 with unpaired values
        unpaired_values = scatter(filtered_v_, batch, dim=0, reduce='max')
        persistence1_new = torch.zeros(
                edge_index.shape[1], filtered_v_.shape[1], 2, device=x.device)
        
        #edge_slices = torch.Tensor(edge_slices).to(x.device)
        edge_slices = edge_slices.to(x.device) 
        bs = edge_slices.shape[0] - 1
        n_edges = edge_slices[1:] - edge_slices[:-1]
        random_edges = (
                edge_slices[0:-1].unsqueeze(-1) +
                torch.floor(
                    torch.rand(size=(bs, self.num_filtrations), device=x.device)
                    * n_edges.float().unsqueeze(-1)
                )
            ).long()

        persistence1_new[random_edges, torch.arange(self.num_filtrations).unsqueeze(0), :] = (
                torch.stack([
                    unpaired_values,
                    filtered_e_[
                        random_edges, torch.arange(self.num_filtrations).unsqueeze(0)]
                ], -1)
            )
        return persistence0_new.permute(1, 0, 2), persistence1_new.permute(1, 0, 2)


    def forward(self, x, edge_index, batch, vertex_slices, edge_slices):

        if self.fake:
            pers0, pers1 = self.compute_fake_persistence(
            x, edge_index, vertex_slices, edge_slices, batch
            )
        else:
            pers0, pers1 = self.compute_persistence(
            x, edge_index, vertex_slices, edge_slices, batch
            )


        x0 = torch.cat(
            [x, pers0.permute(1, 0, 2).reshape(pers0.shape[1], -1)], 1)

        if self.dim1_flag:
        # Dim 1 computations.
            pers1_reshaped = pers1.permute(1,0,2).reshape(pers1.shape[1],-1)
            pers1_mask = ~((pers1_reshaped==0).all(-1))
            nodes_idx_dim1 = edge_index[:,pers1_mask]
            x0_dim1 = torch.cat(
                [ x[nodes_idx_dim1[0,:],:], x[nodes_idx_dim1[1,:],:], pers1_reshaped[pers1_mask]  ], 1)
            x_dim1 = self.dim1_fn(x0_dim1, edge_slices, mask = pers1_mask)
        else:
            x_dim1 = None
        
        
        for layer in self.set_fn0:
            if isinstance(layer, DeepSetLayer):
                x0 = layer(x0, batch)
            else:
                x0 = layer(x0)

        # Collect valid
        # valid_0 = (pers1 != 0).all(-1)

        return x + self.bn(x0), x_dim1


class FakeSetTopoLayer(nn.Module):
    def __init__(self, n_features, n_filtrations, mlp_hidden_dim, aggregation_fn):
        super().__init__()
        self.filtrations = nn.Sequential(
            nn.Linear(n_features, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, n_filtrations),
        )
        self.set_fn0 = nn.ModuleList([
            DeepSetLayer(n_features + n_filtrations,
                         mlp_hidden_dim, aggregation_fn),
            nn.ReLU(),
            DeepSetLayer(mlp_hidden_dim, n_features, aggregation_fn),
        ])
        self.bn = nn.BatchNorm1d(n_features)
        # self.set_fn1 = nn.ModuleList([
        #     DeepSetLayer(n_filtrations*2, mlp_hidden_dim),
        #     nn.ReLU(),
        #     DeepSetLayer(mlp_hidden_dim, n_features),
        # ])

    def forward(self, x, edge_index, batch, vertex_slices, edge_slices):
        filtered_v = self.filtrations(x)

        x0 = torch.cat([x, filtered_v], 1)
        for layer in self.set_fn0:
            if isinstance(layer, DeepSetLayer):
                x0 = layer(x0, batch)
            else:
                x0 = layer(x0)

        # Collect valid
        # valid_0 = (pers1 != 0).all(-1)

        return x + self.bn(x0), None
