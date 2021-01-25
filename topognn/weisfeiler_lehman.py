"""Transformer for Weisfeiler—Lehman Feature Vector calculations."""

# This code was originally used in the paper 'A Persistent Weisfeiler–Lehman
# Procedure for Graph Classification'. The original author is Christian Bock.

import collections
import copy

import numpy as np


class WeisfeilerLehman:
    """Implement Weisfeiler–Lehman feature vector generation."""

    def __init__(self):
        """Create new instance of class."""
        self._relabel_steps = collections.defaultdict(dict)
        self._label_dict = {}
        self._last_new_label = -1
        self._preprocess_relabel_dict = {}
        self._results = collections.defaultdict(dict)
        self._label_dicts = {}

    def _reset_label_generation(self):
        self._last_new_label = -1

    def _get_next_label(self):
        self._last_new_label += 1
        return self._last_new_label

    def _relabel_graphs(self, X):
        """Auxiliary function for relabelling a graph."""
        preprocessed_graphs = []
        for i, g in enumerate(X):
            x = g.copy()
            labels = x.vs['label']

            new_labels = []
            for label in labels:
                if label in self._preprocess_relabel_dict.keys():
                    new_labels.append(self._preprocess_relabel_dict[label])
                else:
                    self._preprocess_relabel_dict[label] = \
                            self._get_next_label()

                    new_labels.append(self._preprocess_relabel_dict[label])

            x.vs['label'] = new_labels
            self._results[0][i] = (labels, new_labels)

            preprocessed_graphs.append(x)

        self._reset_label_generation()
        return preprocessed_graphs

    def fit_transform(self, X, num_iterations=3):
        """Perform transformation of input list of graphs."""
        X = self._relabel_graphs(X)
        for it in np.arange(1, num_iterations+1, 1):
            self._reset_label_generation()
            self._label_dict = {}
            for i, g in enumerate(X):
                # Get labels of current interation
                current_labels = g.vs['label']

                # Get for each vertex the labels of its neighbors
                neighbor_labels = self._get_neighbor_labels(g, sort=True)

                # Prepend the vertex label to the list of labels of its
                # neighbors.
                merged_labels = [
                    [b]+a for a, b in zip(neighbor_labels, current_labels)
                ]

                # Generate a label dictionary based on the merged labels
                self._append_label_dict(merged_labels)

                # Relabel the graph
                new_labels = self._relabel_graph(g, merged_labels)

                self._relabel_steps[i][it] = {
                    idx: {
                        old_label: new_labels[idx]
                    } for idx, old_label in enumerate(current_labels)
                }

                g.vs['label'] = new_labels

                self._results[it][i] = (merged_labels, new_labels)
            self._label_dicts[it] = copy.deepcopy(self._label_dict)
        return self._results

    def _relabel_graph(self, X, merged_labels):
        """Extend graph with new merged labels."""
        new_labels = []
        for merged in merged_labels:
            new_labels.append(self._label_dict['-'.join(map(str, merged))])
        return new_labels

    def _append_label_dict(self, merged_labels):
        for merged_label in merged_labels:
            dict_key = '-'.join(map(str, merged_label))
            if dict_key not in self._label_dict.keys():
                self._label_dict[dict_key] = self._get_next_label()

    def _get_neighbor_labels(self, X, sort=True):
        neighbor_indices = [
            [n_v.index for n_v in X.vs[X.neighbors(v.index)]] for v in X.vs
        ]
        neighbor_labels = []
        for n_indices in neighbor_indices:
            if sort:
                neighbor_labels.append(sorted(X.vs[n_indices]['label']))
            else:
                neighbor_labels.append(X.vs[n_indices]['label'])

        return neighbor_labels
