"""Create output CSV files for paper."""

import argparse

import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input file')

    args = parser.parse_args()

    df = pd.read_csv(args.INPUT)

    columns_to_keep = [
        'Name',
        'val_acc',
        'test_acc',
        'depth',
        'dim1',
    ]

    df = df[columns_to_keep]
    df = df.fillna(0)

    print(df.to_csv(index=False))
