"""Analyse set of graphs using Weisfeiler--Lehman feature iteration."""

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

from weisfeiler_lehman import WeisfeilerLehman


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

    parser.add_argument('INPUT', type=str, nargs='+', help='Input file(s)')
    parser.add_argument(
        '-H', '--num-iterations',
        default=3,
        type=int,
        help='Number of iterations for the Weisfeiler--Lehman algorithm'
    )
    parser.add_argument(
        '-p', '--pickle',
        action='store_true',
        help='If set, loads graphs from pickle file'
    )
    parser.add_argument(
        '-l', '--labels',
        type=str,
        help='Path to labels'
    )

    args = parser.parse_args()
    H = args.num_iterations

    # Will contain all graphs in `igraph` format. They will form the
    # basis for the analysis in terms of Weisfeiler--Lehman features
    # later on.
    graphs = []

    # Ditto for labels, but this is optional.
    labels = []

    if args.labels is not None:
        labels = torch.load(args.labels).numpy()

    for filename in args.INPUT:
        if args.pickle:
            with open(filename, 'rb') as f:
                x_list, edge_lists = pickle.load(f)

                for edge_list in edge_lists:
                    graphs.append(build_graph_from_edge_list(edge_list))
        else:
            g = ig.Graph.Read_Edgelist(filename, directed=False)
            g.vs['label'] = g.degree()

            graphs.append(g)

    # The idea is to show that the feature vectors are the same between
    # two distributions of graphs (or require more steps than warranted
    # as the cycle length increases).

    wl = WeisfeilerLehman()
    label_dicts = wl.fit_transform(graphs, num_iterations=H)

    # Will contain the feature matrix. Rows are indexing individual
    # graphs, columns are indexing all iterations of the scheme, so
    # that the full WL iteration is contained in one vector.
    X = []

    for i, g in enumerate(graphs):

        # All feature vectors of the current graph
        x = []
        for h in range(H):
            _, compressed_labels = label_dicts[h][i]
            x.extend(np.bincount(compressed_labels).tolist())

        X.append(x)

    # Ensure that all feature vectors have the same length.
    L = 0
    for x in X:
        L = max(L, len(x))

    X = [x + [0] * (L - len(x)) for x in X]
    X = np.asarray(X)

    # Norm distribution of all vectors; not sure whether this will be
    # useful.
    norms = np.sqrt(np.sum(np.abs(X)**2, axis=-1))
    print(f'Norm distribution of WL feature vectors: {norms}')

    distances = euclidean_distances(X)
    print(f'Mean distance between WL feature vectors: {np.mean(distances)}')

    if args.labels is None:
        sys.exit(0)

    print('Fitting cross-validated classifier on data...')

    scores = []

    for i in range(10):
        cv = StratifiedKFold(n_splits=5, shuffle=True)

        param_grid = {
            'C': 10. ** np.arange(-3, 4),  # 10^{-3}..10^{3}
        }

        # This follows the original WL paper as closely as possible. The
        # goal is *not* to obtain the highest performance but to show to
        # what extent deeper iterations help in classifying the graphs.
        svm = SVC(kernel='linear')
        clf = GridSearchCV(svm, param_grid, scoring='accuracy')
        clf.fit(X, labels)

        scores.append(np.mean(cross_val_score(clf, X, labels, cv=cv)))
        print(f'Iteration {i}: {100 * scores[-1]:.2f}')

    print(f'{100 * np.mean(scores):.2f} +- {100 * np.std(scores):.2f}')
