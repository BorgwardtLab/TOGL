"""Plot a pair of graphs."""

import argparse
import pickle
import torch
import sys

import igraph as ig
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.svm import SVC


def build_graph_from_edge_list(edge_list):
    """Build graph from edge list and return it."""
    n_vertices = edge_list.max().numpy() + 1
    g = ig.Graph(n_vertices)

    for u, v in edge_list.numpy().transpose():
        g.add_edge(u, v)

    g.vs['label'] = g.degree()
    return g


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('INPUT', type=str, help='Input file')
    parser.add_argument(
        '-l', '--labels',
        type=str,
        help='Path to labels'
    )

    parser.add_argument('FIRST', type=int, help='First graph to plot')
    parser.add_argument('SECOND', type=int, help='Second graph to plot')

    args = parser.parse_args()

    # Will contain all graphs in `igraph` format. They will form the
    # basis for the plotting later.
    graphs = []

    # Ditto for labels, but we just use them to tell us about the ID
    # of a graph (if specified by the user).
    labels = []

    if args.labels is not None:
        labels = torch.load(args.labels).numpy()

    with open(args.INPUT, 'rb') as f:
        x_list, edge_lists = pickle.load(f)

        for edge_list in edge_lists:
            graphs.append(build_graph_from_edge_list(edge_list))

    g1 = graphs[args.FIRST]
    g2 = graphs[args.SECOND]

    print(f'Plotting graph {args.FIRST} and {args.SECOND}')

    if len(labels) != 0:
        print(f'  Labels: {labels[args.FIRST]}/{labels[args.SECOND]}')

    for g in [g1, g2]:
        layout = g.layout('kk')
        ig.plot(g, layout=layout)
