import numpy as np
from er_kmeans import ERKMeans


class ERKMeansSafe:
    """
    Wraps ERKMeans with standard multi-restart practice (cf. sklearn KMeans'
    n_init): if a fit collapses (any cluster falls below min_cluster_frac of
    the data, or fewer than k clusters survive), the run is discarded and
    retried with a fresh random initialization. If no restart succeeds, it
    falls back to lambda=0 (plain K-Means), which can never collapse a
    cluster below its own baseline behavior.

    This is the required practical usage pattern for entropy-regularized
    assignment at lambda values large enough to meaningfully affect balance;
    without it, a single unlucky initialization can degrade badly.
    """

    def __init__(self, k, lambda_reg, max_restarts=10, min_cluster_frac=0.01,
                 max_iters=100, random_state=0):
        self.k = k
        self.lambda_reg = lambda_reg
        self.max_restarts = max_restarts
        self.min_cluster_frac = min_cluster_frac
        self.max_iters = max_iters
        self.random_state = random_state

    def fit(self, X):
        n = X.shape[0]
        rng = np.random.RandomState(self.random_state)

        for attempt in range(self.max_restarts):
            seed = int(rng.randint(0, 2**31 - 1))
            model = ERKMeans(k=self.k, lambda_reg=self.lambda_reg,
                              max_iters=self.max_iters, random_state=seed)
            model.fit(X)
            frac_min = model.cluster_sizes_.min() / n
            n_unique = len(np.unique(model.labels_))
            if frac_min >= self.min_cluster_frac and n_unique == self.k:
                self.labels_ = model.labels_
                self.centroids_ = model.centroids_
                self.cluster_sizes_ = model.cluster_sizes_
                self.n_restarts_used_ = attempt + 1
                self.fell_back_ = False
                return self

        # Fallback: plain K-Means (lambda=0) never collapses below its own
        # natural cluster structure in the same pathological way.
        model = ERKMeans(k=self.k, lambda_reg=0.0, max_iters=self.max_iters,
                          random_state=self.random_state)
        model.fit(X)
        self.labels_ = model.labels_
        self.centroids_ = model.centroids_
        self.cluster_sizes_ = model.cluster_sizes_
        self.n_restarts_used_ = self.max_restarts
        self.fell_back_ = True
        return self

    def get_balance_metrics(self):
        n = sum(self.cluster_sizes_)
        k = self.k
        p = self.cluster_sizes_ / n
        entropy = -np.sum(p * np.log(p + 1e-12))
        h_norm = entropy / np.log(k) if np.log(k) > 0 else 0
        return {'h_norm': h_norm, 'entropy': entropy, 'sizes': self.cluster_sizes_.tolist()}
