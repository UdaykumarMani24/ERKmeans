#!/usr/bin/env python3
"""
Final data loader reproducing the exact 13-dataset battery in Table 1
of the manuscript. Real datasets: Iris, Wine, Breast Cancer, Digits
(sklearn-bundled) + HTRU2 (real pulsar data, fetched once and cached
locally as htru2_clean.csv). Synthetic datasets: the 8 CSVs already
published in the GitHub repo's synthetic_datasets/ folder, loaded
directly so results are reproducible from the released artifacts.
"""
import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler

REPO_SYNTH_DIR = "/home/claude/work/repo/synthetic_datasets"
HTRU2_PATH = "/home/claude/work/htru2_clean.csv"


def _scale(X):
    return StandardScaler().fit_transform(X)


def load_real_datasets():
    out = {}

    d = load_iris()
    out['Iris'] = (_scale(d.data.astype(float)), d.target, 3)

    d = load_wine()
    out['Wine'] = (_scale(d.data.astype(float)), d.target, 3)

    d = load_breast_cancer()
    out['Breast Cancer'] = (_scale(d.data.astype(float)), d.target, 2)

    d = load_digits()
    out['Digits'] = (_scale(d.data.astype(float)), d.target, 10)

    htru = pd.read_csv(HTRU2_PATH, header=None)
    X = htru.iloc[:, :-1].values.astype(float)
    y = htru.iloc[:, -1].values.astype(int)
    out['HTRU2'] = (_scale(X), y, 2)

    return out


def load_synthetic_datasets():
    mapping = {
        'syn_3c_60': ('syn_3c_imbalance_60.csv', 3),
        'syn_3c_70': ('syn_3c_imbalance_70.csv', 3),
        'syn_3c_80': ('syn_3c_imbalance_80.csv', 3),
        'syn_3c_90': ('syn_3c_imbalance_90.csv', 3),
        'syn_2c_95': ('syn_2c_imbalance_95.csv', 2),
        'syn_5c_10k_70': ('syn_5c_n10000_imb70.csv', 5),
        'syn_5c_10k_90': ('syn_5c_n10000_imb90.csv', 5),
        'syn_10c_50k_80': ('syn_10c_n50000_imb80.csv', 10),
    }
    out = {}
    for name, (fname, k) in mapping.items():
        df = pd.read_csv(os.path.join(REPO_SYNTH_DIR, fname))
        X = df[['feature_1', 'feature_2']].values.astype(float)
        y = df['label'].values.astype(int)
        out[name] = (X, y, k)
    return out


def load_all():
    """Returns dict: name -> (X, y, k), in the same order as Table 1."""
    out = {}
    out.update(load_real_datasets())
    out.update(load_synthetic_datasets())
    return out


if __name__ == "__main__":
    all_ds = load_all()
    for name, (X, y, k) in all_ds.items():
        sizes = np.bincount(y, minlength=k)
        print(f"{name:18s} n={X.shape[0]:6d} d={X.shape[1]:3d} k={k:2d} sizes={sizes.tolist()}")
