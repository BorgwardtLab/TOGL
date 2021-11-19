"""Analyse set of graphs using simple degree-based filtration."""

import argparse
import pickle
import torch

import igraph as ig
import numpy as np

from pyper.persistent_homology.graphs import calculate_persistence_diagrams
from pyper.persistent_homology.graphs import extend_filtration_to_edges
from pyper.vectorisation import featurise_distances
from pyper.vectorisation import featurise_pairwise_distances

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVC


def build_graph_from_edge_list(x, edge_list):
    """Build graph from edge list and return it."""
    n_vertices = edge_list.max().numpy() + 1
    g = ig.Graph(n_vertices)

    x = x.numpy()
    x = np.linalg.norm(x, axis=1)

    for u, v in edge_list.numpy().transpose():
        g.add_edge(u, v)

    if args.random:
        g.vs['attribute'] = np.random.normal(size=len(g.degree()))
    elif args.norm_filtration:
        g.vs['attribute'] = x
    else:
        g.vs['attribute'] = g.degree()

    if args.perturb:
        g.vs['attribute'] += np.random.normal(size=len(g.degree()))

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

    parser.add_argument(
        '-r', '--random',
        action='store_true',
        help='If set, uses random node features instead of degrees.'
    )

    parser.add_argument(
        '-p', '--perturb',
        action='store_true',
        help='If set, perturbs filtration values.'
    )

    parser.add_argument(
        '-N', '--norm-filtration',
        action='store_true',
        help='If set, use norm-based filtration.'
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

            for x, edge_list in zip(x_list, edge_lists):
                graphs.append(build_graph_from_edge_list(x, edge_list))

    persistence_diagrams = [
        calculate_persistence_diagrams(
            graph,
            vertex_attribute='attribute',
            edge_attribute='attribute',
            unpaired=100,
        ) for graph in graphs
    ]

    X0 = []
    X1 = []

    for (D0, D1) in persistence_diagrams:
        x0 = featurise_distances(D0)
        x1 = featurise_distances(D1)

        # Simple padding...this one might bite us at some point.
        x0 += [0] * (50 - len(x0))
        x1 += [0] * (50 - len(x1))

        X0.append(x0)
        X1.append(x1)

    X0 = np.asarray(X0)
    X1 = np.asarray(X1)
    X = np.hstack((X0, X1))

    scores = []

    for i in range(10):
        cv = StratifiedKFold(n_splits=3, shuffle=True)

        # Could also be done in a stratified fashion...
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        param_grid = {
            'C': 10. ** np.arange(-3, 4),  # 10^{-3}..10^{3}
            'gamma': ['auto'],
        }

        svm = SVC(kernel='rbf')
        clf = GridSearchCV(svm, param_grid, scoring='accuracy', cv=3)
        clf.fit(X, labels)

        scores.append(np.mean(cross_val_score(clf, X, labels, cv=cv)))
        print(f'Iteration {i}: {100 * scores[-1]:.2f}')

    print(f'{100 * np.mean(scores):.2f} +- {100 * np.std(scores):.2f}')
