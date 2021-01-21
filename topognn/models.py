import math
import torch
import torch.nn.functional as F

import pytorch_lightning as pl

from torch_geometric.nn import GCNConv, global_mean_pool

from torch_geometric.data import DataLoader, Batch, Data

from topognn.topo_utils import batch_persistence_routine, persistence_routine, parallel_persistence_routine

import topognn.coord_transforms as coord_transforms
import numpy as np


class TopologyLayer(torch.nn.Module):
    """Topological Aggregation Layer."""

    def __init__(self, features_in, features_out, num_filtrations, num_coord_funs, filtration_hidden, num_coord_funs1=None, dim1=False):
        """
        num_coord_funs is a dictionary with the numbers of coordinate functions of each type.
        dim1 is a boolean. True if we have to return dim1 persistence.
        """
        super().__init__()

        self.dim1 = dim1

        self.features_in = features_in
        self.features_out = features_out

        self.num_filtrations = num_filtrations
        self.num_coord_funs = num_coord_funs

        self.filtration_hidden = filtration_hidden

        self.total_num_coord_funs = np.array(
            list(num_coord_funs.values())).sum()

        self.coord_fun_modules = torch.nn.ModuleList([
            getattr(coord_transforms, key)(output_dim=num_coord_funs[key])
            for key in num_coord_funs
        ])

        if self.dim1:
            assert num_coord_funs1 is not None
            self.coord_fun_modules1 = torch.nn.ModuleList([
                getattr(coord_transforms, key)(output_dim=num_coord_funs1[key])
                for key in num_coord_funs1
            ])

        # TODO : do we want to do weight sharing ? e.g. one big NN with num_filtrations outputs.
        self.filtration_modules = torch.nn.ModuleList([

            torch.nn.Sequential(
                torch.nn.Linear(self.features_in, self.filtration_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(self.filtration_hidden, 1)) for _ in range(num_filtrations)
        ])

        self.out = torch.nn.Linear(
            self.features_in + self.num_filtrations * self.total_num_coord_funs, features_out)

    def compute_persistence(self, x, batch):
        """
        Returns the persistence pairs as a list of tensors with shape [X.shape[0],2].
        The lenght of the list is the number of filtrations.
        """
        filtered_v_ = torch.cat([filtration_mod.forward(x)
                                 for filtration_mod in self.filtration_modules], 1)

        filtered_v_cpu = filtered_v_.cpu()

        # TEST
        #persistence0 = parallel_persistence_routine(filtered_v_cpu, batch).to(filtered_v_.device)
        #persistence0 = torch.split(persistence0,1,2)
        #persistence0 = [p.squeeze(-1) for p in persistence0]

        persistence0 = []
        persistence1 = []

        for f_idx in range(self.num_filtrations):

            batch_cpu = batch.clone().to("cpu")
            batch_p_ = batch_persistence_routine(
                filtered_v_cpu[:, f_idx], batch_cpu, self.dim1)

            if self.dim1:  # cycles were computed
                persistence0.append(batch_p_[0].to(filtered_v_.device))
                persistence1.append(batch_p_[1].to(filtered_v_.device))
            else:
                persistence0.append(batch_p_.to(filtered_v_.device))

        return persistence0, persistence1

    def compute_coord_fun(self, persistence, dim1=False):
        """
        Input : persistence [N_points,2]
        Output : coord_fun mean-aggregated [self.num_coord_fun]
        """
        if dim1:
            coord_activation = torch.cat(
                [mod.forward(persistence) for mod in self.coord_fun_modules1], 1)
        else:
            coord_activation = torch.cat(
                [mod.forward(persistence) for mod in self.coord_fun_modules], 1)

        return coord_activation

    def compute_coord_activations(self, persistences, batch, dim1=False):
        """
        Return the coordinate functions activations pooled by graph.
        Output dims : list of length number of filtrations with elements : [N_graphs in batch, number fo coordinate functions]
        """

        coord_activations = [self.compute_coord_fun(
            persistence, dim1=dim1) for persistence in persistences]

        return torch.cat(coord_activations, 1)

    def collapse_dim1(self, activations, mask, slices):
        """
        #TODO: @edward Comment this
        """
        collapsed_activations = []
        for el in range(len(slices)-1):
            activations_el_ = activations[slices[el]:slices[el+1]]
            mask_el = mask[slices[el]:slices[el+1]]
            activations_el = activations_el_[mask_el].sum(axis=0)
            collapsed_activations.append(activations_el)

        return torch.stack(collapsed_activations)

    def forward(self, x, batch):

        persistences0, persistences1 = self.compute_persistence(x, batch)

        coord_activations = self.compute_coord_activations(
            persistences0, batch)

        concat_activations = torch.cat((x, coord_activations), 1)

        out_activations = self.out(concat_activations)

        if self.dim1:
            persistence1_mask = (persistences1[0] == 0).sum(axis=1) == 2
            # TODO potential save here by only computing the activation on the masked persistences
            coord_activations1 = self.compute_coord_activations(
                persistences1, batch, dim1=True)

            graph_activations1 = self.collapse_dim1(coord_activations1, persistence1_mask, batch.__slices__[
                                                    "edge_index"])  # returns a vector for each graph
        else:
            graph_activations1 = None

        return out_activations, graph_activations1


class FiltrationGCNModel(pl.LightningModule):
    """
    GCN Model with a Graph Filtration Readout function.
    """

    def __init__(self, hidden_dim, filtration_hidden, num_node_features, num_classes, num_filtrations, num_coord_funs, dim1=False, num_coord_funs1=None, lr=0.001, dropout_p=0.2):
        """
        num_filtrations = number of filtration functions
        num_coord_funs = number of different coordinate function
        """
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.topo1 = TopologyLayer(hidden_dim, hidden_dim, num_filtrations=num_filtrations,
                                   num_coord_funs=num_coord_funs, filtration_hidden=filtration_hidden, dim1=dim1, num_coord_funs1=num_coord_funs1)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.dim1 = dim1
        # number of extra dimension for each embedding from cycles (dim1)
        if dim1:
            cycles_dim = num_filtrations * \
                np.array(list(num_coord_funs1.values())).sum()
        else:
            cycles_dim = 0

        self.classif = torch.nn.Sequential(torch.nn.Linear(hidden_dim+cycles_dim, hidden_dim),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(hidden_dim, num_classes))

        self.loss = torch.nn.CrossEntropyLoss()

        self.num_filtrations = num_filtrations
        self.num_coord_funs = num_coord_funs

        self.accuracy = pl.metrics.Accuracy()
        self.accuracy_val = pl.metrics.Accuracy()
        self.accuracy_test = pl.metrics.Accuracy()

        self.lr = lr
        self.dropout_p = dropout_p

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        #x = self.conv2(x, edge_index)
        #x = F.relu(x)
        #x = F.dropout(x, training = self.training)

        x, x_dim1 = self.topo1(x, data)

        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, data.batch)

        if self.dim1:
            x_pre_class = torch.cat([x, x_dim1], axis=1)
        else:
            x_pre_class = x

        x_out = self.classif(x_pre_class)

        return x_out

    def training_step(self, batch, batch_idx):
        y = batch.y

        y_hat = self(batch)

        loss = self.loss(y_hat, y)

        self.accuracy(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", self.accuracy, on_step=True, on_epoch=True)
        return loss

    # def training_epoch_end(self,outs):
    #    self.log("train_acc_epoch",self.accuracy.compute())

    def validation_step(self, batch, batch_idx):
        y = batch.y
        y_hat = self(batch)

        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, on_epoch=True)

        self.accuracy_val(y_hat, y)
        self.log("val_acc", self.accuracy_val, on_epoch=True)

        return {"predictions": y_hat.detach().cpu(),
                "labels": y.detach().cpu()}

    def test_step(self, batch, batch_idx):
        y = batch.y
        y_hat = self(batch)

        loss = self.loss(y_hat, y)
        self.log("test_loss", loss, on_epoch=True)

        self.accuracy_test(y_hat, y)
        self.log("test_acc", self.accuracy_test, on_epoch=True)


class GCNModel(pl.LightningModule):
    def __init__(self, hidden_dim, num_node_features, num_classes, lr=0.001, dropout_p=0.2):
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.fake_topo = torch.nn.Linear(hidden_dim, hidden_dim)

        self.classif = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(hidden_dim, num_classes))

        self.loss = torch.nn.CrossEntropyLoss()

        self.accuracy = pl.metrics.Accuracy()
        self.accuracy_val = pl.metrics.Accuracy()
        self.accuracy_test = pl.metrics.Accuracy()

        self.lr = lr

        self.dropout_p = dropout_p

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.fake_topo(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, data.batch)

        x_out = self.classif(x)

        return x

    def training_step(self, batch, batch_idx):
        y = batch.y
        y_hat = self(batch)

        loss = self.loss(y_hat, y)

        self.accuracy(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", self.accuracy, on_step=True, on_epoch=True)
        return loss

    # def training_epoch_end(self,outs):
    #    self.log("train_acc_epoch",self.accuracy.compute())

    def validation_step(self, batch, batch_idx):
        y = batch.y
        y_hat = self(batch)

        loss = self.loss(y_hat, y)

        self.accuracy_val(y_hat, y)
        self.log("val_loss", loss)

        self.log("val_acc", self.accuracy_val, on_epoch=True)

        return {"predictions": y_hat.detach().cpu(),
                "labels": y.detach().cpu()}

    def test_step(self, batch, batch_idx):

        y = batch.y
        y_hat = self(batch)

        loss = self.loss(y_hat,y)
        self.log("test_loss",loss, on_epoch = True)
        
        self.accuracy_test(y_hat, y)

        self.log("test_acc", self.accuracy_test, on_epoch=True)

    # def validation_epoch_end(self,outputs):
    #    self.log("val_acc_epoch", self.accuracy_val.compute())
