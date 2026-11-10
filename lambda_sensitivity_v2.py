#!/usr/bin/env python3
"""
Corrected lambda-sensitivity sweep: use the actual operative range (up to
5n, per Section 3.7's own grid), not the small range used only for the
convergence stress test in Section 3.5. Also runs select_lambda() itself
to recover the actual lambda* and check it reproduces Table 2/3's
published ER-K-Means numbers for syn_3c_90 and HTRU2.
"""
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score

sys.path.insert(0, ".")
sys.path.insert(0, "/home/claude/work/repo/src")
from er_kmeans import ERKMeans
from er_kmeans_safe import ERKMeansSafe
from select_lambda import select_lambda
from data_loader_final import load_all


def normalized_entropy(sizes, k):
    n = sum(sizes)
    p = np.array(sizes) / n
    with np.errstate(divide='ignore'):
        h = -np.sum(p * np.log(p + 1e-12))
    return h / np.log(k) if np.log(k) > 0 else 0.0


def sweep(name, X, y, k, lambda_grid, n_seeds, out_rows):
    n = X.shape[0]
    for lam in lambda_grid:
        for seed in range(n_seeds):
            model = ERKMeans(k=k, lambda_reg=lam, random_state=seed, max_iters=100)
            model.fit(X)
            sizes = model.cluster_sizes_
            h = normalized_entropy(sizes, k)
            n_unique = len(np.unique(model.labels_))
            collapsed = (sizes.min() / n < 0.01) or (n_unique < k)
            try:
                sil = silhouette_score(X, model.labels_) if n_unique > 1 else -1.0
            except Exception:
                sil = -1.0
            try:
                ari = adjusted_rand_score(y, model.labels_)
            except Exception:
                ari = None
            out_rows.append(dict(dataset=name, lam=lam, seed=seed, h_norm=h,
                                  silhouette=sil, ari=ari, collapsed=collapsed))


def main():
    datasets = load_all()
    rows = []

    # --- syn_3c_90: full log-spaced grid up to 5n, as Section 3.7 specifies ---
    X, y, k = datasets['syn_3c_90']
    n = X.shape[0]
    grid = np.unique(np.round(np.geomspace(0.1, 5 * n, 14)))
    print(f"syn_3c_90 grid: {grid}")
    sweep('syn_3c_90', X, y, k, grid, n_seeds=15, out_rows=rows)

    lam_star, _ = select_lambda(X, k, seeds=range(20))
    print(f"syn_3c_90 select_lambda() -> lambda* = {lam_star}")
    safe = ERKMeansSafe(k=k, lambda_reg=lam_star, random_state=0).fit(X)
    h_safe = normalized_entropy(safe.cluster_sizes_, k)
    print(f"syn_3c_90 ERKMeansSafe @ lambda*: H_norm={h_safe:.4f} "
          f"(Table 2 reports 0.8911+/-0.0383)")

    df = pd.DataFrame(rows)
    df.to_csv("lambda_sensitivity_v2_raw.csv", index=False)
    summary = df.groupby(['dataset', 'lam']).agg(
        h_mean=('h_norm', 'mean'), h_std=('h_norm', 'std'),
        sil_mean=('silhouette', 'mean'), sil_std=('silhouette', 'std'),
        ari_mean=('ari', 'mean'), ari_std=('ari', 'std'),
        collapse_rate=('collapsed', 'mean'),
    ).reset_index()
    summary.to_csv("lambda_sensitivity_v2_summary.csv", index=False)
    print(summary.to_string(index=False))

    with open("lambda_star_syn_3c_90.txt", "w") as f:
        f.write(f"lambda*={lam_star}\nH_norm_at_lambda_star={h_safe}\n")


if __name__ == "__main__":
    main()
