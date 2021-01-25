"""Analyse set of graphs using Weisfeiler--Lehman feature iteration."""

import argparse

import igraph as ig
import numpy as np


from sklearn.metrics.pairwise import euclidean_distances
from weisfeiler_lehman import WeisfeilerLehman


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('INPUT', type=str, nargs='+', help='Input file(s)')
    parser.add_argument(
        '-H', '--num-iterations',
        default=3,
        type=int,
        help='Number of iterations for the Weisfeiler--Lehman algorithm'
    )

    args = parser.parse_args()
    H = args.num_iterations

    # Will contain all graphs in `igraph` format. They will form the
    # basis for the analysis in terms of Weisfeiler--Lehman features
    # later on.
    graphs = []

    for filename in args.INPUT:
        g = ig.Graph.Read_Edgelist(filename, directed=False)
        g.vs['label'] = g.degree()

        graphs.append(g)

    # First analysis step: degree distribution
    #
    # The idea is to obtain a mean degree distribution that does not
    # depend on the number of cycles.
    X = []

    for g in graphs:
        degrees = g.degree()
        X.append(np.bincount(degrees))

    X = np.asarray(X)

    # Second analysis step: Weisfeiler--Lehman feature vectors
    #
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
    print(norms)

    distances = euclidean_distances(X)
    print(np.mean(distances))
