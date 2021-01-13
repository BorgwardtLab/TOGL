import torch
from pyper.persistent_homology.graphs import calculate_persistence_diagrams
from pyper.utilities import UnionFind
from torch_geometric.data import Batch, Data

def batch_persistence_routine_old(filtered_v_, batch):

    batch_persistence = [persistence_routine(filtered_v_[batch.batch==i],data) for i,data in enumerate(batch.to_data_list())]

    return torch.cat(batch_persistence)

def batch_persistence_routine(filtered_v_, batch):
    """
    Persistence diagrams are computed in one shot. 
    Note that this results in a tiny but potentially significant difference compared to computing the persistence one graph at a time.
    Namely, the unpaired_value (inf) will be set to the largest value in the *batch* rather than to the highest value in the *graph*.
    """
    return persistence_routine(filtered_v_, batch)

def persistence_routine(filtered_v_, data: Data, method = "new", cycles = False):
    """
    Pytorch based routine to compute the persistence pairs
    Based on pyper routine.
    Inputs : 
        * filtration values of the vertices
        * data object that stores the graph structure (could be just the edge_index actually)
        * method is just a check for the algo
        * cycles is a boolean to compute the 1D persistence or not. If true, returns also the 1D persistence.
    """
    
    #Quick check for the filtration values to be different.
    #if torch.unique(filtered_v_).reshape(-1,1).shape[0] != filtered_v_.reshape(-1,1).shape[0]:
        #if not unique, we add a small perturbation on all the values with std 0.01 x the initial std of the filtration values.
        #std = torch.std(filtered_v_)
        #filtered_v_ += 0.001*std*torch.randn(filtered_v_.shape)

    # Compute the edge filtrations as the max between the value of the nodes.
    filtered_e_, _ = torch.max(torch.stack((filtered_v_[data.edge_index[0]],filtered_v_[data.edge_index[1]])),axis=0)

    # Only the edges need to be sorted, since they determine the
    # filtration ordering. For the vertices, we will look up the
    # values directly in the tensor.
    filtered_e, e_indices = torch.sort(filtered_e_)

    n_vertices = len(filtered_v_)
    uf = UnionFind(n_vertices)

    persistence = torch.zeros(
        (n_vertices, 2),
        device = filtered_v_.device
    )

    edge_indices_cycles =  []

    for edge_index, edge_weight in zip(e_indices,filtered_e):
      
        # nodes connected to this edge
        nodes = data.edge_index[:, edge_index]

        younger = uf.find(nodes[0])
        older = uf.find(nodes[1])

        if younger == older :
            if cycles:
                edge_indices_cycles.append(edge_index)
            continue
        else:
            # Use vertex weight lookup to determine which vertex comes
            # first. This works because our filtrations are based on
            # values at the vertices themselves.
            if filtered_v_[younger] < filtered_v_[older]:
                younger, older = older, younger
                nodes = torch.flip(nodes, [0])

        # TODO: can this be removed? 
        #persistence[nodes[0],0] = filtered_v_[younger]
        #persistence[nodes[0],1] = edge_weight

        persistence[younger, 0] = filtered_v_[younger]
        persistence[younger, 1] = edge_weight
        
        uf.merge(nodes[0],nodes[1])
    
    #TODO : this currently assumes a single unpaired value for the whole batch. THis can be discussed.
    unpaired_value = filtered_e[-1]
    for root in uf.roots():
        persistence[root,0] = filtered_v_[root]
        persistence[root,1] = unpaired_value

    if cycles:
        persistence1 = torch.zeros((len(filtered_e),2)) 
        for edge_index in enumerate(edge_indices_cycles):
            persistence1[i,0] = filtered_e_[edge_index]
            persistence1[i,1] = unpaired_value
        return persistence, persistence1

    return persistence


#numpy / pyper stuff 
import igraph as ig


def batch_to_igraph_list(batch: Batch):
    list_of_instances = batch.to_data_list()

    graphs = [ig.Graph(zip(instance.edge_index[0].tolist(),instance.edge_index[1].tolist())) for instance in list_of_instances]
    
    return graphs




def apply_degree_filtration(graphs):
    for graph in graphs:
        graph.vs['filtration'] = graph.vs.degree()

        for edge in graph.es:
            u, v = edge.source, edge.target
            edge['filtration'] = max(graph.vs[u]['filtration'],
                                     graph.vs[v]['filtration'])

    return graphs

def calculate_batch_persistence_diagrams(graphs):
    persistence_diagrams = [
        calculate_persistence_diagrams(
            graph, vertex_attribute='filtration', edge_attribute='filtration')
        for graph in graphs
    ]
    return persistence_diagrams


def persistence_diagrams_from_batch(batch: Batch):
    graphs = batch_to_igraph_list(batch)
    graphs = apply_degree_filtration(graphs)
    return calculate_batch_persistence_diagrams(graphs)

