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

    parser.add_argument('INPUT', type=str, help='Input file')
    parser.add_argument(
        '-l', '--labels',
        type=str,
        help='Path to labels'
    )
    parser.add_argument(
        '-H', '--num-iterations',
        default=3,
        type=int,
        help='Number of iterations for the Weisfeiler--Lehman algorithm'
    )

    parser.add_argument('FIRST', type=int, help='First graph to plot')
    parser.add_argument('SECOND', type=int, help='Second graph to plot')

    args = parser.parse_args()
    H = args.num_iterations

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

    wl = WeisfeilerLehman()
    label_dicts = wl.fit_transform([g1, g2], num_iterations=H)

    for i, g in enumerate([g1, g2]):
        for h in label_dicts:

            # We only want the compressed labels here.
            _, labels = label_dicts[h][i]
            g.vs['label'] = labels

            print(
                f'Graph {i + 1} @ iteration {h + 1}: ',
                np.bincount(labels)
            )

            layout = g.layout('kk')
            ig.plot(g, layout=layout, target=f'/tmp/G_{h + 1}_{i}.png')

            print('\\begin{tikzpicture}')

            for j, (x, y) in enumerate(layout.coords):
                print(f'  \\coordinate ({j:02d}) at ({x:.2f}, {y:.2f});')

            for e in g.es:
                u, v = e.source, e.target
                print(f'  \\draw ({u:02d}) -- ({v:02d});')

            s = '  \\foreach \\v in {'
            for j in range(len(layout.coords)):
                s += f'{j:02d}'
                s += ',' if j + 1 < len(layout.coords) else ''
            s += '}'

            print(s)
            print('  {')
            print('    \\filldraw (\\v) circle (1pt);')
            print('  }')

            print('\\end{tikzpicture}')
