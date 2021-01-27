"""Convert adjacency matrix format to edge list."""

import argparse
import os

import numpy as np


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

    A = np.loadtxt(args.INPUT)
    out = os.path.join(
        args.output,
        os.path.splitext(os.path.basename(args.INPUT))[0] + '.txt'
    )

    with open(out, 'w') as f:
        for u, v in np.transpose(np.nonzero(A != 0)):
            if u < v:
                print(u, v, file=f)
