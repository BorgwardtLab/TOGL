import torch
from pyper.persistent_homology.graphs import calculate_persistence_diagrams
#from pyper.utilities import UnionFind

from torch_geometric.data import Batch, Data
#from topognn.unionfind_torch import UnionFind

import unionfind

import time
import numpy as np


def batch_persistence_routine_old(filtered_v_, batch):

    batch_persistence = [persistence_routine(
        filtered_v_[batch.batch == i], data) for i, data in enumerate(batch.to_data_list())]

    return torch.cat(batch_persistence)


def batch_persistence_routine(filtered_v_, batch, dim1=False):
    """
    Persistence diagrams are computed in one shot. 
    Note that this results in a tiny but potentially significant difference compared to computing the persistence one graph at a time.
    Namely, the unpaired_value (inf) will be set to the largest value in the *batch* rather than to the highest value in the *graph*.
    """
    return persistence_routine(filtered_v_, batch, cycles=dim1)


def persistence_routine(filtered_v_, data: Data, cycles=False):
    """
    Pytorch based routine to compute the persistence pairs
    Based on pyper routine.
    Inputs : 
        * filtration values of the vertices
        * data object that stores the graph structure (could be just the edge_index actually)
        * method is just a check for the algo
        * cycles is a boolean to compute the 1D persistence or not. If true, returns also the 1D persistence.
    """

    # Quick check for the filtration values to be different.
    # if torch.unique(filtered_v_).reshape(-1,1).shape[0] != filtered_v_.reshape(-1,1).shape[0]:
    # if not unique, we add a small perturbation on all the values with std 0.01 x the initial std of the filtration values.
    #std = torch.std(filtered_v_)
    #filtered_v_ += 0.001*std*torch.randn(filtered_v_.shape)

    # Compute the edge filtrations as the max between the value of the nodes.
    start_time = time.time()

    filtered_e_, _ = torch.max(torch.stack(
        (filtered_v_[data.edge_index[0]], filtered_v_[data.edge_index[1]])), axis=0)


    # Only the edges need to be sorted, since they determine the
    # filtration ordering. For the vertices, we will look up the
    # values directly in the tensor.
    filtered_e, e_indices = torch.sort(filtered_e_)

    n_vertices = len(filtered_v_)

    #uf = UnionFind(n_vertices)
    uf = unionfind.UnionFind(n_vertices)  # Cython version

    persistence = torch.zeros(
        (n_vertices, 2),
        device=filtered_v_.device
    )
    if cycles:
        persistence1 = torch.zeros(
            (len(filtered_e), 2), device=filtered_v_.device)

    edge_indices_cycles = []

    pre_time = time.time()

    unpaired_value = filtered_e[-1]

    persistence[:, 0] = filtered_v_

    for edge_index, edge_weight in zip(e_indices, filtered_e):

        # nodes connected to this edge
        nodes = data.edge_index[:, edge_index]
        
        younger = uf.find(nodes[0])
        older = uf.find(nodes[1])

        if younger == older:
            if cycles:
                # edge_indices_cycles.append(edge_index)
                persistence1[edge_index, 0] = filtered_e_[edge_index]
                persistence1[edge_index, 1] = unpaired_value
            continue
        else:
            # Use vertex weight lookup to determine which vertex comes
            # first. This works because our filtrations are based on
            # values at the vertices themselves.
            if filtered_v_[younger] < filtered_v_[older]:
                younger, older = older, younger
                nodes = torch.flip(nodes, [0])

        #persistence[younger, 0] = filtered_v_[younger]
        persistence[younger, 1] = edge_weight

        uf.merge(nodes[0], nodes[1])

    loop_time = time.time()

    # TODO : this currently assumes a single unpaired value for the whole batch. THis can be discussed.
    for root in uf.roots():
        persistence[root, 0] = filtered_v_[root]
        persistence[root, 1] = unpaired_value

    end_time = time.time()
    # if cycles:
    #persistence1 = torch.zeros((len(filtered_e),2), device = filtered_v_.device)
    # for edge_index in edge_indices_cycles:
    #    persistence1[edge_index,0] = filtered_e_[edge_index]
    #    persistence1[edge_index,1] = unpaired_value

    #cycle_time = time.time()

    if cycles:
        return persistence, persistence1
    else:
        return persistence


def parallel_persistence_routine(filtered_v_, data: Data, cycles=False):
    """
    Pytorch based routine to compute the persistence pairs in parallel !
    Based on pyper routine.
    Inputs : 
        * filtration values of the vertices shape is [N,D] where D is the number of filtrations
        * data object that stores the graph structure (could be just the edge_index actually)
        * cycles is a boolean to compute the 1D persistence or not. If true, returns also the 1D persistence.
    """

    # Quick check for the filtration values to be different.
    # if torch.unique(filtered_v_).reshape(-1,1).shape[0] != filtered_v_.reshape(-1,1).shape[0]:
    # if not unique, we add a small perturbation on all the values with std 0.01 x the initial std of the filtration values.
    #std = torch.std(filtered_v_)
    #filtered_v_ += 0.001*std*torch.randn(filtered_v_.shape)

    # Compute the edge filtrations as the max between the value of the nodes.
    start_time = time.time()

    num_filtrations = filtered_v_.shape[1]

    filtered_e_, _ = torch.max(torch.stack(
        (filtered_v_[data.edge_index[0]], filtered_v_[data.edge_index[1]])), axis=0)

    # Only the edges need to be sorted, since they determine the
    # filtration ordering. For the vertices, we will look up the
    # values directly in the tensor.
    filtered_e, e_indices = torch.sort(filtered_e_, 0)

    n_vertices = len(filtered_v_)
    #uf = UnionFind(n_vertices)
    uf = unionfind.MultipleUnionFind(num_filtrations, n_vertices)

    persistence = torch.zeros(
        (n_vertices, 2, num_filtrations),
        device=filtered_v_.device
    )

    if cycles:
        persistence1 = torch.zeros(
            (len(filtered_e), 2, num_filtrations), device=filtered_v_.device)

    edge_indices_cycles = []

    pre_time = time.time()

    unpaired_values = filtered_e[-1]

    for edge_index, edge_weight in zip(e_indices, filtered_e):

        # nodes connected to this edge
        nodes = data.edge_index[:, edge_index]

        younger = np.array(uf.find(nodes[0]))
        older = np.array(uf.find(nodes[1]))

        values_younger = filtered_v_[younger, np.arange(num_filtrations)]
        values_older = filtered_v_[older, np.arange(num_filtrations)]

        equal_mask = values_younger == values_older
        correct_mask = values_younger > values_older
        flipped_mask = values_younger < values_older

        persistence[younger[correct_mask], 0, np.arange(
            num_filtrations)[correct_mask]] = values_younger[correct_mask]
        persistence[older[flipped_mask], 0, np.arange(
            num_filtrations)[flipped_mask]] = values_older[flipped_mask]

        persistence[younger[correct_mask], 1, np.arange(
            num_filtrations)[correct_mask]] = edge_weight[correct_mask]
        persistence[older[flipped_mask], 1, np.arange(
            num_filtrations)[flipped_mask]] = edge_weight[flipped_mask]

        if cycles:
            persistence1[edge_index[equal_mask], 0, np.arange(num_filtrations)[equal_mask]] = filtered_e_[
                edge_index, np.arange(num_filtrations)][equal_mask]
            persistence1[edge_index[equal_mask], 1, np.arange(
                num_filtrations)[equal_mask]] = unpaired_values[equal_mask]

        nodes0_index = (correct_mask*1 + 1*equal_mask + flipped_mask*2)-1
        nodes1_index = 1-nodes0_index

        uf.merge(nodes[nodes0_index, np.arange(num_filtrations)],
                 nodes[nodes1_index, np.arange(num_filtrations)], ~equal_mask)

    loop_time = time.time()

    # TODO : this currently assumes a single unpaired value for the whole batch. THis can be discussed.
    for root in uf.roots():
        cycle_mask = np.array(root) != -1
        persistence[root, 0, np.arange(num_filtrations)][cycle_mask] = filtered_v_[
            root, np.arange(num_filtrations)][cycle_mask]
        persistence[root, 1, np.arange(
            num_filtrations)][cycle_mask] = unpaired_values[cycle_mask]

    end_time = time.time()

    if cycles:
        return persistence, persistence1
    else:
        return persistence

# numpy / pyper stuff

#import igraph as ig


def batch_to_igraph_list(batch: Batch):
    list_of_instances = batch.to_data_list()

    graphs = [ig.Graph(zip(instance.edge_index[0].tolist(
    ), instance.edge_index[1].tolist())) for instance in list_of_instances]

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
