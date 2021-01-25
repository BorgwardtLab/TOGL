"""Convert files from `graph6` format to edge lists."""

import argparse
import math
import os

import networkx as nx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('INPUT', help='Input file', type=str)

    parser.add_argument(
        '-o', '--output',
        type=str,
        default='/tmp',
        help='Output directory'
    )

    args = parser.parse_args()

    graphs = nx.read_graph6(args.INPUT)

    n = len(graphs)
    n_digits = int(math.log10(n) + 1)

    for i, g in enumerate(graphs):
        out = os.path.join(args.output, f'G_{i:0{n_digits}d}.txt')
        nx.write_edgelist(g, out, data=False)
