"""
Adaptive lambda selection: sweep a log-spaced grid, and pick the largest
lambda BEFORE balance collapses toward degenerate solutions (i.e. before
h_norm starts dropping sharply from its local maximum / before any cluster
empties out). This formalizes "how to choose lambda" instead of guessing.
"""
import numpy as np
from er_kmeans import ERKMeans

def select_lambda(X, k, lambda_grid=None, seeds=range(20), min_cluster_frac=0.01,
                   max_degenerate_rate=0.5):
    """
    Sweep a log-spaced grid (scaled to n, since the correction ~ lambda/n).
    For each lambda, measure both the mean balance among NON-degenerate runs
    and the degenerate (collapse) rate. Select the lambda that maximizes
    balance among non-degenerate outcomes, subject to the collapse rate
    being low enough (<= max_degenerate_rate) that the ERKMeansSafe
    multi-restart wrapper will reliably recover a valid fit in a handful of
    attempts. This intentionally allows some per-restart collapse risk,
    since ERKMeansSafe (standard multi-restart practice, cf. sklearn's
    n_init) discards degenerate restarts rather than needing lambda itself
    to guarantee zero collapse risk.
    """
    n = X.shape[0]
    if lambda_grid is None:
        lambda_grid = np.unique(np.round(np.geomspace(0.1, 5*n, 60)))

    records = []
    for lam in lambda_grid:
        h_vals, degenerate = [], []
        for seed in seeds:
            erk = ERKMeans(k=k, lambda_reg=lam, random_state=seed, max_iters=100)
            erk.fit(X)
            sizes = erk.cluster_sizes_
            frac_min = sizes.min() / n
            is_degenerate = (frac_min < min_cluster_frac) or (len(np.unique(erk.labels_)) < k)
            degenerate.append(is_degenerate)
            if not is_degenerate:
                h_vals.append(erk.get_balance_metrics()['h_norm'])
        records.append(dict(lam=lam,
                             h_mean_nondegenerate=np.mean(h_vals) if h_vals else np.nan,
                             degenerate_rate=np.mean(degenerate)))

    candidates = [r for r in records
                  if r['degenerate_rate'] <= max_degenerate_rate and not np.isnan(r['h_mean_nondegenerate'])]
    if not candidates:
        return 0.0, records
    best = max(candidates, key=lambda r: r['h_mean_nondegenerate'])
    return best['lam'], records
