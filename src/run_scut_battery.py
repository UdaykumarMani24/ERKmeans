#!/usr/bin/env python3
"""
Run Scut (Malinen & Franti, 2023) across the full 13-dataset battery,
matching the manuscript's exact protocol: 15-20 seeds for n<=2000,
8-10 seeds for the larger-scale synthetic sets (Section 4.3).
"""
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score

sys.path.insert(0, ".")
from data_loader_final import load_all
from scut import Scut

N_RUNS_SMALL = 20   # n <= 2000
N_RUNS_LARGE = 10   # n > 2000


def normalized_entropy(sizes, k):
    n = sum(sizes)
    p = np.array(sizes) / n
    with np.errstate(divide='ignore'):
        h = -np.sum(p * np.log(p + 1e-12))
    return h / np.log(k) if np.log(k) > 0 else 0.0


def main():
    datasets = load_all()
    rows = []

    for name, (X, y, k) in datasets.items():
        n_runs = N_RUNS_SMALL if X.shape[0] <= 2000 else N_RUNS_LARGE
        print(f"\n{name} (n={X.shape[0]}, k={k}): {n_runs} runs", flush=True)
        for seed in range(n_runs):
            t0 = time.time()
            model = Scut(k=k, random_state=seed).fit(X)
            elapsed = time.time() - t0

            sizes = model.cluster_sizes_
            h_norm = normalized_entropy(sizes, k)
            try:
                sil = silhouette_score(X, model.labels_)
            except Exception:
                sil = -1.0
            try:
                ari = adjusted_rand_score(y, model.labels_)
            except Exception:
                ari = None

            rows.append(dict(dataset=name, method='Scut', seed=seed,
                              h_norm=h_norm, silhouette=sil, ari=ari,
                              time=elapsed, sizes=str(sizes.tolist())))
            print(f"  seed={seed:2d} h_norm={h_norm:.4f} sil={sil:.4f} "
                  f"ari={ari if ari is not None else 'NA'} t={elapsed:.2f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("scut_results.csv", index=False)

    print("\n" + "=" * 70)
    print("SCUT SUMMARY (mean +/- std)")
    print("=" * 70)
    summary = df.groupby('dataset').agg(
        h_mean=('h_norm', 'mean'), h_std=('h_norm', 'std'),
        sil_mean=('silhouette', 'mean'), sil_std=('silhouette', 'std'),
        ari_mean=('ari', 'mean'), ari_std=('ari', 'std'),
        time_mean=('time', 'mean'),
    )
    print(summary.to_string())
    summary.to_csv("scut_summary.csv")


if __name__ == "__main__":
    main()
