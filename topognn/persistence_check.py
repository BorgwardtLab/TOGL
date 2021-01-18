from gtda.homology import VietorisRipsPersistence
import numpy as np 
import copy
import torch
from torch_geometric.data import Batch

from topognn.topo_utils import persistence_routine

import topognn.data_utils as topodata
from pyper.persistent_homology.graphs import calculate_persistence_diagrams

import igraph as ig

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

if __name__ =="__main__":

    #Test 0 :
    print("Test 0 (from the Graph Filtration paper) ")
    edge_index = np.array([[0,0,2,3,4],[2,3,3,4,1]])
    filtered_v = np.array([1.,1.,2.,3.,4.])
    dim0, dim1 = compute_persistence_giotto(edge_index, filtered_v)
   
    batch = Batch()
    batch.edge_index = torch.tensor(edge_index)
    dim0_torch, dim1_torch = persistence_routine(torch.tensor(filtered_v),batch,cycles = True) 

    persistence_pyper, persistence_pyper1 = compute_pyper_persistence(edge_index, filtered_v) 

    assert pers_to_set(persistence_pyper) == pers_to_set(dim0_torch)
    assert pers_to_set(filter_persistence(dim1_torch)) == pers_to_set(persistence_pyper1)

    #---------------------------
    #---------- Test 1 ---------
    #---------------------------
    edge_index = np.array(torch.load("edge_index.pt"))
    filtered_v = np.array(torch.load("filtered_v_.pt"))
    filtered_v += 0.01*np.random.randn(len(filtered_v))

    # --------- Giotto ---------
    persistence_giotto = compute_persistence_giotto(edge_index, filtered_v)[0]

    print("-----Giotto----")
    print(persistence_giotto)

    # --------- TopoGNN --------
    batch = Batch()
    batch.edge_index = torch.tensor(edge_index)
    persistence_torch, persistence_torch1 = persistence_routine(torch.tensor(filtered_v), batch, cycles = True)
    
    print("------TopoGNN------")
    print(persistence_torch)

    # --------- Pyper ---------
    persistence_pyper, persistence_pyper1 = compute_pyper_persistence(edge_index, filtered_v)

    print("------Pyper-----")
    print(persistence_pyper)

    assert pers_to_set(persistence_pyper) == pers_to_set(persistence_torch)
    assert pers_to_set(persistence_giotto) == pers_to_set(filter_persistence(persistence_torch))

    assert pers_to_set(filter_persistence(persistence_pyper1)) == pers_to_set(filter_persistence(persistence_torch1))


    #-------------------------
    #--------- Test2 ---------
    #-------------------------

    #Check the batch version of torch persistence.

    data = topodata.TUGraphDataset('ENZYMES', batch_size=32)
    data.prepare_data()
    batch = next(iter(data.train_dataloader()))
    batch.x = torch.mean(batch.x,axis=1)
    batch_list = batch.to_data_list()

    #batch computation
    persistence_torch, persistence_torch1 = persistence_routine(batch.x, batch, cycles = True)
    batch_persistence = copy.copy(batch)
    batch_persistence.x = persistence_torch
    persistences_batch = [ b.x for b in batch_persistence.to_data_list()]

    #graph computation
    persistences_single = [persistence_routine(b.x,b,cycles = True)[0] for b in batch_list]
   
    for ib in range(32):
        a = pers_to_set(persistences_single[ib]) 
        b = pers_to_set(persistences_batch[ib])
        assert len(b-a) == len(a-b)
        assert {a_[0] for a_ in a} == {b_[0] for b_ in b}

