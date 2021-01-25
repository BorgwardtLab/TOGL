"""Create graphs with a pre-defined number of cycles."""

import argparse

import igraph as ig
import numpy as np


def make_vertex_chain(start, n, edges):
    """Create chain of vertices, starting from a given index."""
    v = start

    for i in range(n):
        edges.append((v, v + 1))
        v += 1

    return edges, v


def make_cycle_graph(
    n_cycles,
    min_length,
    max_length,
    n_pre=10,
    n_mid=5,
    n_post=10
):
    """Create random graph with pre-defined number of cycles."""
    # No edges and no vertices by default. This will be updated by the
    # cycle creation procedure.
    v = 0
    edges = []

    edges, v = make_vertex_chain(v, np.random.randint(2, n_pre + 1), edges)

    for i in range(n_cycles):
        cycle_len = np.random.randint(min_length, max_length + 1)

        v_start = v
        edges, v = make_vertex_chain(v, cycle_len - 1, edges)
        edges.append((v, v_start))

        edges, v = make_vertex_chain(v, np.random.randint(2, n_mid + 1), edges)

    edges, v = make_vertex_chain(v, np.random.randint(2, n_post + 1), edges)
    return edges, v + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n', '--n-cycles',
        type=int,
        help='Number of cycles',
        default=3,
    )

    parser.add_argument(
        '-m', '--n-graphs',
        type=int,
        help='Number of graphs to generate',
        default=100,
    )

    parser.add_argument(
        '-l', '--min-length',
        type=int,
        help='Minimum length of cycle',
        default=3,
    )

    parser.add_argument(
        '-L', '--max-length',
        type=int,
        help='Maximum length of cycle',
        default=10,
    )

    args = parser.parse_args()

    # Will contain all graphs in `igraph` format. They will form the
    # basis for the analysis in terms of Weisfeiler--Lehman features
    # later on.
    graphs = []

    for i in range(args.n_graphs):
        edges, n_vertices = make_cycle_graph(
            args.n_cycles,
            args.min_length,
            args.max_length
        )

        g = ig.Graph()
        g.add_vertices(n_vertices)
        g.add_edges(edges)

        graphs.append(g)

        # FIXME: Also store the graph for persistent homology
        # calculations. Need to decide on a format here.
        with open(f'/tmp/G_{args.n_cycles}_{i:02d}.txt', 'w') as f:
            for u, v in edges:
                print(f'{u} {v}', file=f)
