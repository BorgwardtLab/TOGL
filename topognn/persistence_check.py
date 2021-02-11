#!/usr/bin/env python
from gtda.homology import VietorisRipsPersistence
import numpy as np 
import copy
import torch
from torch_geometric.data import Batch, Data

from topognn.topo_utils import persistence_routine

import topognn.data_utils as topodata
from pyper.persistent_homology.graphs import calculate_persistence_diagrams


from torch_persistent_homology.persistent_homology_cpu import compute_persistence_homology_batched_mt


import igraph as ig

import time


def compute_persistence_torch(edge_index, filtered_v_, batch):
        """
        Persistence computation with torch_persistence_computation
        """
        filtered_e_, _ = torch.max(torch.stack(
            (filtered_v_[edge_index[0]], filtered_v_[edge_index[1]])), axis=0)

        vertex_slices = torch.Tensor(batch.__slices__['x']).cpu().long()
        edge_slices = torch.Tensor(batch.__slices__['edge_index']).cpu().long()

        filtered_v_ = filtered_v_.cpu().transpose(1, 0).contiguous()
        filtered_e_ = filtered_e_.cpu().transpose(1, 0).contiguous()
        edge_index = edge_index.cpu().transpose(1, 0).contiguous()

        persistence0_new, persistence1_new = compute_persistence_homology_batched_mt(
            filtered_v_, filtered_e_, edge_index,
            vertex_slices, edge_slices)

        return persistence0_new, persistence1_new


def compute_persistence_giotto(edge_index, filtered_v):
    adj = np.zeros((len(filtered_v),len(filtered_v)))

    edge_filtrations = np.max(np.stack((filtered_v[edge_index[0,:]],filtered_v[edge_index[1,:]])),axis = 0)

    for i,edge in enumerate(edge_index.T):
        adj[edge[0],edge[1]] = edge_filtrations[i]
        adj[edge[1],edge[0]] = edge_filtrations[i]

    adj[adj==0] = np.inf

    np.fill_diagonal(adj,filtered_v)

    VR = VietorisRipsPersistence(homology_dimensions=[0,1], metric = "precomputed",infinity_values=np.max(filtered_v),reduced_homology = False)

    VR = VR.fit(adj[None,:,:])

    persistence = VR.transform(adj[None,:,:])

    dim0 = persistence[persistence[:,:,-1]==0,:-1]
    dim1 = persistence[persistence[:,:,-1]==1,:-1]


    return dim0, dim1

def compute_pyper_persistence(edge_index, filtered_v):

    edge_tuples = list(zip(list(edge_index[0]),list(edge_index[1])))
    filtered_e = np.max(np.stack((filtered_v[edge_index[0]],filtered_v[edge_index[1]])),axis=0)

    g = ig.Graph(list(zip(list(edge_index[0]),list(edge_index[1]))),
                   edge_attrs={"filtration":filtered_e})
    g.vs["filtration"] = filtered_v
    persistence_pyper, persistence_pyper1 = calculate_persistence_diagrams(
            g, vertex_attribute='filtration', edge_attribute='filtration')#, order = "superlevel")

    return torch.tensor(np.array(persistence_pyper)).float(), torch.tensor(np.array(persistence_pyper1)).float()

def pers_to_set(input_tensor):
    """
    Transforms a persistence tensor to a set.
    Useful for comparison with pyper and giotto that returns non ordered persistences.
    """
    return set(tuple(x) for x in input_tensor.tolist())

def filter_persistence(input_tensor):
    return input_tensor[input_tensor[:,0]!=input_tensor[:,1]]

def remove_duplicate_edges(batch):
    flipped_idx = batch.edge_index[0] > batch.edge_index[1]
    batch.edge_index[:,flipped_idx] = torch.flip(batch.edge_index[:,flipped_idx],dims=(0,))
    batch.edge_index = torch.unique(batch.edge_index,dim=1)

def remove_duplicate_edges_bis(batch):
    correct_idx = batch.edge_index[0] <= batch.edge_index[1]
    batch.edge_index = batch.edge_index[:,correct_idx]



if __name__ =="__main__":

    #Test 0 :
    print("Test 0 (from the Graph Filtration paper) ")
    edge_index = np.array([[0,1,3,2,3],[3,2,2,4,4]])
    #edge_index = np.concatenate((edge_index,edge_index[[1,0],:]),axis=1)

    filtered_v = np.array([1.,2.,3.,4.,5.])
    dim0, dim1 = compute_persistence_giotto(edge_index, filtered_v)
   
    data = Data()    
    data.edge_index = torch.tensor(edge_index)
    data.x = torch.tensor(filtered_v)
    batch = Batch().from_data_list([data])

    dim0_torch, dim1_torch = persistence_routine(torch.tensor(filtered_v),batch,cycles = True)

    dim0_torch_new, dim1_torch_new = compute_persistence_torch(torch.tensor(edge_index), torch.tensor(filtered_v[:,None]), batch)
    
    persistence_pyper, persistence_pyper1 = compute_pyper_persistence(edge_index, filtered_v) 

    assert pers_to_set(persistence_pyper) == pers_to_set(dim0_torch)
    assert pers_to_set(filter_persistence(dim1_torch)) == pers_to_set(persistence_pyper1)
    assert (dim0_torch == dim0_torch_new).all()
    assert (dim1_torch == dim1_torch_new).all()

    import ipdb; ipdb.set_trace()
         

    #Test 0 bis :
    print("Test 0bis (arbitrary graph) ")
    edge_index = np.array([[0,1,2,3,2,2,5,8,6,6,7,9],[1,2,3,4,7,5,6,7,8,9,8,10]])
    filtered_v = np.array([1.,2.,4.,2.,2.,2.,4.,2.,2.,2.,1.])
    
    persistence_pyper, persistence_pyper1 = compute_pyper_persistence(edge_index, filtered_v)     


    edge_index = np.array([[0,1,2,3,2,2,5,6,6,6,7,9],[1,2,3,4,4,5,6,7,8,9,8,10]])
    filtered_v = np.array([1.,2.,4.,2.,2.,2.,4.,2.,2.,2.,1.])
    persistence_pyper_bis, persistence_pyper1_bis = compute_pyper_persistence(edge_index, filtered_v)

    import ipdb; ipdb.set_trace()


    #---------------------------
    #---------- Test 1 ---------
    #---------------------------
    edge_index = np.array(torch.load("./topognn/edge_index.pt"))
    filtered_v = np.array(torch.load("./topognn/filtered_v_.pt"))
    filtered_v += 0.05*np.random.randn(len(filtered_v))

    # --------- Giotto ---------
    persistence_giotto = compute_persistence_giotto(edge_index, filtered_v)[0]

    print("-----Giotto----")
    print(persistence_giotto)

    # --------- TopoGNN --------

    data = Data()    
    data.edge_index = torch.tensor(edge_index)
    data.x = torch.tensor(filtered_v)
    batch = Batch().from_data_list([data])

    persistence_torch, persistence_torch1 = persistence_routine(torch.tensor(filtered_v), batch, cycles = True)
    
    print("------TopoGNN------")
    print(persistence_torch)

    # --------- Pyper ---------
    persistence_pyper, persistence_pyper1 = compute_pyper_persistence(edge_index, filtered_v)

    print("------Pyper-----")
    print(persistence_pyper)

    #----- new persistence torch
    rnd_gen = torch.Generator().manual_seed(421)

    noise = torch.zeros_like(filtered_e_)
    noise.random_(generator = rnd_gen)

    dim0_torch_new, dim1_torch_new = compute_persistence_torch(torch.tensor(edge_index), torch.tensor(filtered_v[:,None]), batch)
    

    assert pers_to_set(persistence_pyper) == pers_to_set(persistence_torch)
    assert pers_to_set(persistence_giotto) == pers_to_set(filter_persistence(persistence_torch))

    assert pers_to_set(filter_persistence(persistence_pyper1)) == pers_to_set(filter_persistence(persistence_torch1))

    #import ipdb; ipdb.set_trace()
    assert (persistence_torch == dim0_torch_new).all()
    #assert (persistence_torch1 == dim1_torch_new[0]).all()

    #-------------------------
    #--------- Test2 ---------
    #-------------------------

    #Check the batch version of torch persistence.

    data = topodata.TUGraphDataset('ENZYMES', batch_size=32)
    data.prepare_data()
    batch = next(iter(data.train_dataloader()))
    batch.x = torch.mean(batch.x,axis=1) + torch.randn(batch.x.shape[0])
    batch_list = batch.to_data_list()

    #new torch persistence
    dim0_torch_new, dim1_torch_new = compute_persistence_torch(torch.tensor(batch.edge_index), torch.tensor(batch.x[:,None]), batch)

    
    #batch computation
    persistence_torch, persistence_torch1 = persistence_routine(batch.x, batch, cycles = True)
    batch_persistence = copy.copy(batch)
    batch_persistence.x = persistence_torch
    persistences_batch = [ b.x for b in batch_persistence.to_data_list()]

    x1 = persistence_torch[torch.where(dim0_torch_new[0]!=persistence_torch)[0]]
    x2 = dim0_torch_new[0,torch.where(dim0_torch_new[0]!=persistence_torch)[0]]
    diffs = torch.cat((x1,x2),1)


    #graph computation
    persistences_single = [persistence_routine(b.x,b,cycles = True)[0] for b in batch_list]
    persistences_single_tensor = torch.cat(persistences_single,0)

    persistences_single1 = [persistence_routine(b.x,b,cycles = True)[1] for b in batch_list]
    persistences_single1_tensor = torch.cat(persistences_single1,0)

    import ipdb; ipdb.set_trace() 
    assert (persistences_single_tensor == dim0_torch_new[0]).all()
    assert (persistences_single1_tensor == dim1_torch_new[0]).all()


    for ib in range(32):
        a = pers_to_set(persistences_single[ib]) 
        b = pers_to_set(persistences_batch[ib])
        assert len(b-a) == len(a-b)
        assert {a_[0] for a_ in a} == {b_[0] for b_ in b}

    #---------------------------
    #---------- Test 3 ---------
    #---------------------------

    data = topodata.TUGraphDataset('ENZYMES', batch_size=1)
    data.prepare_data()

    mod = torch.nn.Linear(32,1)
    for i,b in enumerate(data.train_dataloader()):
        print(i)
        b.x = torch.mean(b.x,axis=1) + 0.01*np.random.randn(len(b.x))
        edge_index = b.edge_index
        filtered_v = b.x

        # --------- Giotto ---------
        persistence_giotto = compute_persistence_giotto(edge_index.numpy(), filtered_v.numpy())[0]

        #print("-----Giotto----")
        #print(persistence_giotto)

        # --------- TopoGNN --------
        batch = Batch()
        batch.edge_index = torch.tensor(edge_index)
        persistence_torch, persistence_torch1 = persistence_routine(torch.tensor(filtered_v), b, cycles = True)
    
        #print("------TopoGNN------")
        #print(persistence_torch)

        # --------- Pyper ---------
        persistence_pyper, persistence_pyper1 = compute_pyper_persistence(edge_index.numpy(), filtered_v.numpy())

        #print("------Pyper-----")
        #print(persistence_pyper)

        assert pers_to_set(persistence_pyper) == pers_to_set(persistence_torch)
        #assert pers_to_set(persistence_giotto) == pers_to_set(filter_persistence(persistence_torch))

        assert pers_to_set(filter_persistence(persistence_pyper1)) == pers_to_set(filter_persistence(persistence_torch1))

    #------------Test 3 :--------------------
    #----------------------------------------
    print("Test 3 (Multiple Cycles) ")
    edge_index = np.array([[0,0,1,1,2,2,4],[1,5,2,5,3,4,5]])
    filtered_v = np.array([1.,1.,1.,1.,1.,1.])
    dim0, dim1 = compute_persistence_giotto(edge_index, filtered_v)
   
    batch = Batch()
    batch.edge_index = torch.tensor(edge_index)
    dim0_torch, dim1_torch = persistence_routine(torch.tensor(filtered_v),batch,cycles = True) 

    persistence_pyper, persistence_pyper1 = compute_pyper_persistence(edge_index, filtered_v) 

    #assert pers_to_set(persistence_pyper) == pers_to_set(dim0_torch)
    #assert pers_to_set(filter_persistence(dim1_torch)) == pers_to_set(persistence_pyper1)


    #TEST 4 : performance !

    print("Test 4 (Performance) ")
    data = topodata.TUGraphDataset('ENZYMES', batch_size=32)
    data.prepare_data()
    batch = next(iter(data.train_dataloader()))
    batch.x = torch.mean(batch.x,axis=1)
    batch_list = batch.to_data_list()

    #batch computation
    start_time = time.time()
    for i in range(20):
        persistence_torch, persistence_torch1 = persistence_routine(batch.x, batch, cycles = True)
    
    end_time = time.time()

    print(f"Average time for the persistence computation : {(end_time-start_time)/20}")


