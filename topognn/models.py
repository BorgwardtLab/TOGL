
import math
import torch
import torch.nn.functional as F

import pytorch_lightning as pl

from torch_geometric.nn import GCNConv, global_mean_pool

from torch_geometric.data import DataLoader, Batch, Data

from topognn.topo_utils import batch_persistence_routine, persistence_routine, batch_persistence_routine_old

import topognn.coord_transforms as coord_transforms
import numpy as np

class TopologyLayer(torch.nn.Module):
    """Topological Aggregation Layer."""

    
    def __init__(self, features_in, features_out, num_filtrations, num_coord_funs):
        """
        num_coord_funs is a dictionary with the numbers of coordinate functions of each type.
        """
        super().__init__()

        self.features_in = features_in
        self.features_out = features_out

        self.num_filtrations = num_filtrations
        self.num_coord_funs = num_coord_funs
        
        self.total_num_coord_funs = np.array(list(num_coord_funs.values())).sum()
        
        self.coord_fun_modules = torch.nn.ModuleList([
                getattr(coord_transforms,key)(output_dim=num_coord_funs[key])
                for key in num_coord_funs
                ])
        
        self.filtration = torch.nn.Linear(self.features_in, self.num_filtrations)

        self.out = torch.nn.Linear(
            self.features_in + self.num_filtrations * self.total_num_coord_funs, features_out)

    def compute_persistence(self,x,batch):
        """
        Returns the persistence pairs as a list of tensors with shape [X.shape[0],2].
        The lenght of the list is the number of filtrations.
        """ 
        filtered_v_ = self.filtration(x)
        persistences = [ batch_persistence_routine(filtered_v_[:,f_idx],batch) for f_idx in range(self.num_filtrations) ]

        return persistences

    def compute_coord_fun(self,persistence):
        """
        Input : persistence [N_points,2]
        Output : coord_fun mean-aggregated [self.num_coord_fun]
        """

        coord_activation = torch.cat([mod.forward(persistence) for mod in self.coord_fun_modules],1)

        return coord_activation


    def compute_coord_activations(self,persistences,batch):

        """
        Return the coordinate functions activations pooled by graph.
        Output dims : list of length number of filtrations with elements : [N_graphs in batch, number fo coordinate functions]
        """
        coord_activations = [self.compute_coord_fun(persistence) for persistence in persistences]

        return torch.cat(coord_activations,1)


    def forward(self,x, batch):

        persistences = self.compute_persistence(x,batch)

        coord_activations = self.compute_coord_activations(persistences,batch)

        concat_activations = torch.cat((x,coord_activations),1)
        
        out_activations = self.out(concat_activations)

        return out_activations


class FiltrationGCNModel(pl.LightningModule):
    """
    GCN Model with a Graph Filtration Readout function.
    """

    def __init__(self, hidden_dim, num_node_features, num_classes, num_filtrations, num_coord_funs):
        """
        num_filtrations = number of filtration functions
        num_coord_funs = number of different coordinate function
        """
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.topo1 = TopologyLayer(hidden_dim, hidden_dim, num_filtrations = num_filtrations,
                                    num_coord_funs = num_coord_funs )
        self.conv3 = GCNConv(hidden_dim, num_classes)

        self.loss = torch.nn.CrossEntropyLoss()

        self.num_filtrations = num_filtrations
        self.num_coord_funs = num_coord_funs

        self.accuracy = pl.metrics.Accuracy()
        self.accuracy_val = pl.metrics.Accuracy()


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

       
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        #x = self.conv2(x, edge_index)
        #x = F.relu(x)
        #x = F.dropout(x, training = self.training)
       
        x = self.topo1(x,data)
        x = self.conv3(x,edge_index)

        x = global_mean_pool(x, data.batch)

        return x 

    def training_step(self, batch, batch_idx):
        y = batch.y
        y_hat = self(batch)

        loss = self.loss(y_hat,y)

        self.log("train_loss",loss, on_step = True, on_epoch = True)
        self.log("train_acc_step",self.accuracy(y_hat,y))
        return loss

    def training_epoch_end(self,outs):
        self.log("train_acc_epoch",self.accuracy.compute())

    def validation_step(self,batch,batch_idx):
        y = batch.y
        y_hat = self(batch)

        loss = self.loss(y_hat,y)
        self.log("val_loss_step",loss)

        self.log("val_acc_step",self.accuracy_val(y_hat,y))
        
        return {"predictions" : y_hat.detach().cpu(), 
                "labels" : y.detach().cpu()}

    def validation_epoch_end(self,outputs):
        self.log("val_acc_epoch", self.accuracy_val.compute())
            

class GCNModel(pl.LightningModule):
    def __init__(self, hidden_dim, num_node_features, num_classes):
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, num_classes)
        self.loss = torch.nn.CrossEntropyLoss()

        self.accuracy = pl.metrics.Accuracy()
        self.accuracy_val = pl.metrics.Accuracy()
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        #x = self.conv2(x, edge_index)

        #x = F.relu(x)
        #x = F.dropout(x, training = self.training)
       
        x = self.conv3(x,edge_index)

        x = global_mean_pool(x, data.batch)
      
        return x

    def training_step(self, batch, batch_idx):
        y = batch.y
        y_hat = self(batch)

        loss = self.loss(y_hat,y)

        self.log("train_loss",loss, on_step = True, on_epoch = True)
        self.log("train_acc_step",self.accuracy(y_hat,y))
        return loss

    def training_epoch_end(self,outs):
        self.log("train_acc_epoch",self.accuracy.compute())

    def validation_step(self,batch,batch_idx):
        y = batch.y
        y_hat = self(batch)

        loss = self.loss(y_hat,y)
        self.log("val_loss_step",loss)

        self.log("val_acc_step",self.accuracy_val(y_hat,y))
        
        return {"predictions" : y_hat.detach().cpu(), 
                "labels" : y.detach().cpu()}

    def validation_epoch_end(self,outputs):
        self.log("val_acc_epoch", self.accuracy_val.compute())


