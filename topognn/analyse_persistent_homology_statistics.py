"""Analyse persistent homology statistics wrt. their expressivity."""

import argparse

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import euclidean_distances


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT')

    parser.add_argument(
        '-c',
        '--column',
        default='n_features',
        type=str,
        help='Column to use for feature calculation'
    )

    args = parser.parse_args()

    df = pd.read_csv(
        args.INPUT,
        index_col='file',
        dtype={
            'n_features': np.int,
            'total_persistence': np.float,
            'infinity_norm': np.float
        }
    )

    X = []
    n_features = 0

    # Figure out how long the feature vectors have to be.
    for name, df_ in df.groupby('name'):
        n_features = max(
            n_features,
            len(df_.sort_values(by='dimension')[args.column].values)
        )

    for name, df_ in df.groupby('name'):
        # This can be seen as an equivalent to the Betti number
        # calculation.
        feature_vector = df_.sort_values(by='dimension')[args.column].values

        feature_vector = np.pad(
            feature_vector,
            [(0, n_features - len(feature_vector))],
            mode='constant'
        )

        X.append(feature_vector)

    X = np.asarray(X)
    D = euclidean_distances(X)
    n = len(X)

    # Number of graph pairs with equal feature vectors, not accounting
    # for the diagonal because every graph is equal to itself.
    n_equal_pairs = (np.triu(D == 0).sum()) - n
    fraction_equal_pairs = n_equal_pairs / (n * (n - 1) // 2)

    print(
        f'{n} graphs, {n_equal_pairs} / {n * (n - 1) // 2:d} pairs '
        f'({100 * fraction_equal_pairs:.2f}%)'
    )
