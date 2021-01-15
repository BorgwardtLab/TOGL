from gtda.homology import VietorisRipsPersistence
import numpy as np 

import torch
from torch_geometric.data import Batch

from topognn.topo_utils import persistence_routine

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

if __name__ =="__main__":
    #Test 0 :
    print("Test 0 (from the Graph Filtration paper) ")
    edge_index = np.array([[0,0,2,3,4],[2,3,3,4,1]])
    filtered_v = np.array([1.,1.,2.,3.,4.])
    dim0, dim1 = compute_persistence_giotto(edge_index, filtered_v)
   
    batch = Batch()
    batch.edge_index = torch.tensor(edge_index)
    dim0_torch, dim1_torch = persistence_routine(torch.tensor(filtered_v),batch,cycles = True)
    

    #---------------------------
    #---------- Test 1 ---------
    #---------------------------

    # --------- Giotto ---------
    edge_index = np.array(torch.load("edge_index.pt"))
    filtered_v = np.array(torch.load("filtered_v_.pt"))
    persistence_giotto = compute_persistence_giotto(edge_index, filtered_v)[0]
    sorted_persistence_giotto = persistence_giotto[np.argsort(persistence_giotto[:,0])]

    print("-----Giotto----")
    print(sorted_persistence_giotto)

    # --------- TopoGNN --------

    batch = Batch() 
    batch.edge_index = torch.tensor(edge_index)
    persistence_torch = persistence_routine(torch.tensor(filtered_v), batch)
        
    reduced_persistence_torch = persistence_torch[persistence_torch[:,0]!=persistence_torch[:,1]].numpy() 
    sorted_reduced_persistence_torch = reduced_persistence_torch[np.argsort(reduced_persistence_torch[:,0])]
    
    print("------TopoGNN------")
    print(sorted_reduced_persistence_torch)

    # --------- Pyper ---------
    edge_tuples = list(zip(list(edge_index[0]),list(edge_index[1])))
    filtered_e = np.max(np.stack((filtered_v[edge_index[0]],filtered_v[edge_index[1]])),axis=0)

    g = ig.Graph(list(zip(list(edge_index[0]),list(edge_index[1]))),
                   edge_attrs={"filtration":filtered_e})
    g.vs["filtration"] = filtered_v
    persistence_pyper,_ = calculate_persistence_diagrams(
            g, vertex_attribute='filtration', edge_attribute='filtration')#, order = "superlevel")

    persistence_pyper = np.array(persistence_pyper)
    reduced_persistence_pyper = persistence_pyper[persistence_pyper[:,0]!=persistence_pyper[:,1]]

    sorted_reduced_persistence_pyper = reduced_persistence_pyper[np.argsort(reduced_persistence_pyper[:,0])]

    print("------Pyper-----")
    print(sorted_reduced_persistence_pyper)

    assert (sorted_reduced_persistence_pyper == sorted_reduced_persistence_torch).all()
    assert (sorted_persistence_giotto == sorted_reduced_persistence_torch).all()
    assert (sorted_reduced_persistence_pyper == sorted_persistence_giotto).all()

