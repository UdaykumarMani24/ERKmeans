#!/usr/bin/env python3
"""
Scut: "All-pairwise squared distances lead to more balanced clustering"
Malinen, M.I. & Franti, P. (2023). Applied Computing and Intelligence, 3(1), 93-115.
https://doi.org/10.3934/aci.2023006

Minimizes Scut = sum_j n_j * TSE_j = sum_j n_j^2 * MSE_j, the all-pairwise
squared-distance cost, via a sequential local-search move (Algorithm 1 in
the paper). The n_j^2 factor penalizes large clusters directly in the
objective, which the authors show yields more balanced partitions than
standard K-Means's total squared error (TSE) without any hard size
constraint. This is a soft-balance baseline, mechanistically distinct from
both the hard-constrained BKM (Malinen & Franti, 2014) and our own entropy
regularizer: it reweights the existing K-Means cost by cluster size rather
than adding a separate penalty/regularization term.
"""
import numpy as np
from sklearn.metrics import pairwise_distances


class Scut:
    """Sequential local search minimizing the Scut / all-pairwise-distance cost."""

    def __init__(self, k=3, max_passes=100, random_state=None):
        self.k = k
        self.max_passes = max_passes
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        rng = np.random.RandomState(self.random_state)

        # K-Means++ init (same scheme used throughout the paper's baselines)
        centroids = self._kmeans_pp_init(X, rng)
        labels = np.argmin(pairwise_distances(X, centroids), axis=1)

        sizes = np.bincount(labels, minlength=self.k).astype(float)
        centroids = np.array([
            X[labels == i].mean(axis=0) if sizes[i] > 0 else centroids[i]
            for i in range(self.k)
        ])
        # TSE_j = sum_{x in C_j} ||x - mu_j||^2
        tse = np.array([
            np.sum((X[labels == i] - centroids[i]) ** 2) if sizes[i] > 0 else 0.0
            for i in range(self.k)
        ])

        order = np.arange(n)
        for _ in range(self.max_passes):
            changed = False
            rng.shuffle(order)
            for idx in order:
                x = X[idx]
                b = labels[idx]
                if sizes[b] <= 1:
                    continue  # never empty a cluster

                # cost of removing x from its current cluster b
                n_b = sizes[b]
                d_b = np.sum((x - centroids[b]) ** 2)
                delta_remove_tse = (n_b / (n_b - 1)) * d_b if n_b > 1 else 0.0

                best_gain = 0.0
                best_a = -1
                for a in range(self.k):
                    if a == b:
                        continue
                    n_a = sizes[a]
                    d_a = np.sum((x - centroids[a]) ** 2)
                    delta_add_tse = (n_a / (n_a + 1)) * d_a

                    # delta_Scut = [TSE_a' * (n_a+1) - TSE_a * n_a] - [TSE_b*n_b - TSE_b'*(n_b-1)]
                    tse_a_new = tse[a] + delta_add_tse
                    tse_b_new = tse[b] - delta_remove_tse
                    scut_delta = (tse_a_new * (n_a + 1) - tse[a] * n_a) - \
                                 (tse[b] * n_b - tse_b_new * (n_b - 1))
                    gain = -scut_delta  # we want Scut to decrease
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_a = a

                if best_a >= 0:
                    a = best_a
                    n_a, n_b = sizes[a], sizes[b]
                    d_a = np.sum((x - centroids[a]) ** 2)
                    d_b = np.sum((x - centroids[b]) ** 2)

                    # update centroids incrementally
                    centroids[a] = (centroids[a] * n_a + x) / (n_a + 1)
                    if n_b - 1 > 0:
                        centroids[b] = (centroids[b] * n_b - x) / (n_b - 1)

                    tse[a] = tse[a] + (n_a / (n_a + 1)) * d_a
                    tse[b] = max(0.0, tse[b] - (n_b / (n_b - 1)) * d_b) if n_b > 1 else 0.0

                    sizes[a] += 1
                    sizes[b] -= 1
                    labels[idx] = a
                    changed = True

            if not changed:
                break

        self.labels_ = labels
        self.cluster_sizes_ = sizes.astype(int)
        self.centroids_ = centroids
        self.n_iter_ = _ + 1
        return self

    def _kmeans_pp_init(self, X, rng):
        n = X.shape[0]
        centroids = [X[rng.choice(n)]]
        for _ in range(1, self.k):
            d2 = pairwise_distances(X, np.array(centroids)).min(axis=1)
            if d2.sum() == 0:
                d2 = np.ones(n)
            p = d2 / d2.sum()
            centroids.append(X[rng.choice(n, p=p)])
        return np.array(centroids)


if __name__ == "__main__":
    # quick sanity check: on a balanced dataset Scut should behave like K-Means;
    # on an imbalanced one it should shift mass toward the small cluster.
    import sys
    sys.path.insert(0, ".")
    rng = np.random.RandomState(0)
    X1 = rng.randn(900, 2) + np.array([5, 0])
    X2 = rng.randn(100, 2) + np.array([-5, 0])
    X = np.vstack([X1, X2])
    y = np.concatenate([np.zeros(900), np.ones(100)])

    sc = Scut(k=2, random_state=0).fit(X)
    print("Scut cluster sizes:", sc.cluster_sizes_)

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
    print("K-Means cluster sizes:", np.bincount(km.labels_))
