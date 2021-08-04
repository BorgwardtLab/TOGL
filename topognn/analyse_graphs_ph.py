"""Analyse set of graphs using simple degree-based filtration."""

import argparse
import pickle
import torch
import sys

import igraph as ig
import numpy as np

from pyper.persistent_homology.graphs import calculate_persistence_diagrams
from pyper.persistent_homology.graphs import extend_filtration_to_edges


def build_graph_from_edge_list(edge_list):
    """Build graph from edge list and return it."""
    n_vertices = edge_list.max().numpy() + 1
    g = ig.Graph(n_vertices)

    for u, v in edge_list.numpy().transpose():
        g.add_edge(u, v)

    g.vs['attribute'] = g.degree()

    g = extend_filtration_to_edges(
        g,
        vertex_attribute='attribute',
        edge_attribute='attribute',
    )

    return g


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('INPUT', type=str, nargs='+', help='Input file(s)')
    parser.add_argument(
        '-l', '--labels',
        type=str,
        help='Path to labels'
    )

    args = parser.parse_args()

    # Will contain all graphs in `igraph` format. They will form the
    # basis for the analysis in terms of a fixed filtration later on
    graphs = []

    # Ditto for labels, but this is optional.
    labels = []

    if args.labels is not None:
        labels = torch.load(args.labels).numpy()

    for filename in args.INPUT:
        with open(filename, 'rb') as f:
            x_list, edge_lists = pickle.load(f)

            for edge_list in edge_lists:
                g = build_graph_from_edge_list(edge_list)
