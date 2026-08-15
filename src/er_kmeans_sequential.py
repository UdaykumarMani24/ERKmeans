#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ER-K-Means: SEQUENTIAL variant.

This implements the assignment rule as actually specified in the
manuscript's Theorem 1 / Section 3.5: points are processed one at a
time, in a fixed order, and the cluster-size counters (and hence the
entropy correction term) are updated immediately after each point is
assigned -- *before* the next point in the pass is scored. This is
required for the descent argument used in the convergence proof.

This differs from er_kmeans.py's ERKMeans class, which assigns all
points simultaneously using cluster sizes frozen from the end of the
previous full pass (a batch/Lloyd's-style update). That batch version
is NOT the algorithm the existing proof covers.
"""

import numpy as np
from sklearn.metrics import pairwise_distances
import warnings

warnings.filterwarnings('ignore')


class ERKMeansSequential:
    """Entropy-Regularized K-Means with true sequential (online-within-pass) assignment."""

    def __init__(self, k=3, lambda_reg=1.0, max_iters=100, tol=1e-4,
                 random_state=None, verbose=False):
        self.k = k
        self.lambda_reg = lambda_reg
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        if random_state is not None:
            np.random.seed(random_state)

    def _kmeans_plus_plus_init(self, X):
        n = X.shape[0]
        centroids = []
        first_idx = np.random.choice(n)
        centroids.append(X[first_idx])
        for _ in range(1, self.k):
            distances = pairwise_distances(X, np.array(centroids)).min(axis=1)
            if distances.sum() == 0:
                distances = np.ones(n)
            probabilities = distances / distances.sum()
            next_idx = np.random.choice(n, p=probabilities)
            centroids.append(X[next_idx])
        return np.array(centroids)

    def _sequential_pass(self, X, order, n):
        """
        One sequential pass over the data in `order`.
        Cluster sizes are updated immediately after each assignment,
        so later points in the same pass see updated counts (matches
        Theorem 1's proof mechanism). Centroids are held fixed for the
        assignment sub-step (as in Lloyd's alternating scheme) but the
        *entropy correction* uses live, within-pass counts.
        """
        labels = -np.ones(n, dtype=int)
        sizes = np.zeros(self.k, dtype=int)  # start each pass from 0
        dist_to_centroids = pairwise_distances(X, self.centroids_)

        for idx in order:
            d = dist_to_centroids[idx].copy()
            for i in range(self.k):
                size_norm = max(sizes[i] / n, 1e-10)
                correction = (self.lambda_reg / n) * np.log(size_norm) if sizes[i] > 0 else \
                             (self.lambda_reg / n) * np.log(1e-10)
                d[i] = d[i] + correction
            best = np.argmin(d)
            labels[idx] = best
            sizes[best] += 1  # <-- immediate update, this is the key theorem requirement

        return labels, sizes

    def fit(self, X):
        X = np.asarray(X)
        n = X.shape[0]
        self.centroids_ = self._kmeans_plus_plus_init(X)
        self.cluster_sizes_ = np.ones(self.k, dtype=int)
        self.n_iter_ = 0
        self.objective_history_ = []

        rng = np.random.RandomState(self.random_state)
        base_order = np.arange(n)

        prev_labels = None
        for iteration in range(self.max_iters):
            self.n_iter_ = iteration + 1
            order = base_order.copy()
            rng.shuffle(order)  # fixed order per Theorem 1 requires *a* fixed order;
                                  # we resample each pass but freeze it within the pass

            new_labels, new_sizes = self._sequential_pass(X, order, n)

            new_centroids = np.zeros_like(self.centroids_)
            for i in range(self.k):
                mask = new_labels == i
                if new_sizes[i] > 0:
                    new_centroids[i] = X[mask].mean(axis=0)
                else:
                    distances = pairwise_distances(X, self.centroids_).min(axis=1)
                    farthest_idx = np.argmax(distances)
                    new_centroids[i] = X[farthest_idx]
                    new_sizes[i] = 1

            objective = self._compute_objective(X, n, new_labels, new_sizes, new_centroids)
            self.objective_history_.append(objective)

            centroid_shift = np.linalg.norm(new_centroids - self.centroids_)
            labels_changed = prev_labels is None or not np.array_equal(new_labels, prev_labels)

            self.centroids_ = new_centroids
            self.cluster_sizes_ = new_sizes
            prev_labels = new_labels
            self.labels_ = new_labels

            if self.verbose:
                print(f"Iter {self.n_iter_}: Obj={objective:.2f}, Shift={centroid_shift:.6f}, "
                      f"Sizes={self.cluster_sizes_}")

            if centroid_shift < self.tol and not labels_changed:
                break

        return self

    def _compute_objective(self, X, n, labels, cluster_sizes, centroids):
        distortion = 0.0
        for i in range(self.k):
            if cluster_sizes[i] > 0:
                cluster_points = X[labels == i]
                distortion += np.sum((cluster_points - centroids[i]) ** 2)
        p = np.maximum(cluster_sizes / n, 1e-12)
        entropy = -np.sum(p * np.log(p))
        return distortion - self.lambda_reg * entropy

    def get_balance_metrics(self):
        n = sum(self.cluster_sizes_)
        k = self.k
        p = self.cluster_sizes_ / n
        entropy = -np.sum(p * np.log(p + 1e-12))
        h_norm = entropy / np.log(k) if np.log(k) > 0 else 0
        return {'h_norm': h_norm, 'entropy': entropy, 'sizes': self.cluster_sizes_.tolist()}
