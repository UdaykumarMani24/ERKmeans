# Entropy-Regularized K-Means (ER-K-Means)

This repository contains the implementation and experiments for the paper:

**"Entropy-Regularized K-Means: A Framework with Characterized Convergence Behavior for Preventing Cluster Collapse"**
*(previously titled "...A Provably Convergent Framework..." — see [CHANGELOG.md](CHANGELOG.md) for why this changed)*

## Authors
Udayakumar Mani (uthay@bioinfo.sastra.edu)
Lavanya Priyadarshini Ramalingam
Senthilkumar Rathinasamy (senthilrathna@sastra.ac.in)
Green Separation Engineering Laboratory, SASTRA Deemed to be University, Thanjavur, Tamil Nadu, 613 401, India

> **Note (revision update):** during peer review, we found and fixed two issues in the code and manuscript that predate this update: an inverted sign in the entropy-correction term (`src/er_kmeans.py`), and an error in the original convergence proof. Both are documented in full in [CHANGELOG.md](CHANGELOG.md). All results below are from the corrected code.

## Repository Structure
```
├── data/                      # Dataset files (synthetic + HTRU2; Iris/Wine/Breast Cancer/Digits load from scikit-learn)
├── figures/                   # Generated figures
│   ├── figure1_assignment_geometry.png
│   └── figure2_dose_response.png
├── results/                   # Experiment output files
│   ├── experiment_results.csv
│   ├── real_experiment_results.csv
│   └── summary_statistics.csv
├── src/                       # Source code
│   ├── er_kmeans.py           # Main ER-K-Means implementation (corrected)
│   ├── er_kmeans_safe.py      # Recommended API: multi-restart wrapper, avoids collapse
│   ├── er_kmeans_sequential.py# Strictly-sequential variant (used to test the convergence proof)
│   ├── select_lambda.py       # Data-driven lambda selection (Algorithm 2 in the paper)
│   ├── data_loader.py         # Dataset loading functions
│   ├── run_experiments.py     # Main entry point
│   └── results/               # Intermediate results
├── CHANGELOG.md                # What changed and why, in detail
└── README.md                   # This file
```

## Requirements
Python 3.8 or higher. Required packages:
```
pip install numpy scikit-learn scipy matplotlib pandas k-means-constrained
```

## Recommended usage

Use `ERKMeansSafe`, not `ERKMeans` directly, unless you have a specific reason not to — a lambda large enough to meaningfully improve balance can, on an unlucky initialization, collapse the fit to fewer than k clusters. `ERKMeansSafe` detects this and retries automatically (the same idea as scikit-learn's `n_init`).

```python
from er_kmeans_safe import ERKMeansSafe
from select_lambda import select_lambda

best_lambda, _ = select_lambda(X, k=3)          # data-driven selection, see paper Section 3.7
model = ERKMeansSafe(k=3, lambda_reg=best_lambda, random_state=0)
model.fit(X)
print(model.get_balance_metrics())
```

## 1. Running Experiments
```
cd src
python run_experiments.py
```
**Note:** `run_experiments.py` as currently committed still uses the old `ERKMeans` class directly with a fixed `lambda_values=[0.5, 1.0, 2.0]` and the old `data_loader.load_all_datasets()` (Iris/Wine/Heart Disease via `fetch_openml`/UCI). It has not yet been updated to use `ERKMeansSafe`, `select_lambda`, or the new dataset battery (Breast Cancer, Digits, HTRU2). Running it today will use the corrected `er_kmeans.py` (so the sign bug is fixed) but will **not** reproduce the exact numbers in the tables below — those came from the individual scripts in the `experiments/` folder of the analysis package accompanying this revision, not from `run_experiments.py`. Updating `run_experiments.py` to match is a good next task before relying on it for a fresh run.

## 2. Run individual experiment scripts

**Note:** `experiments.py` in this repository is currently an empty placeholder file and does not implement the `--dataset`/`--runs` CLI shown in an earlier version of this README — that instruction did not actually work. Until `experiments.py` is implemented, use `run_experiments.py` (above) for the full suite, or call `select_lambda` / `ERKMeansSafe` directly as shown in "Recommended usage" for a single dataset:

```python
from data_loader import generate_synthetic_imbalanced
from select_lambda import select_lambda
from er_kmeans_safe import ERKMeansSafe

X, y = generate_synthetic_imbalanced(n_samples=1000, k=2, imbalance_ratio=0.95, random_state=42)
best_lambda, _ = select_lambda(X, k=2)
model = ERKMeansSafe(k=2, lambda_reg=best_lambda, random_state=0)
model.fit(X)
print(model.get_balance_metrics())
```

## Expected Results

After running the experiments, you should obtain results matching the paper (numbers below are with the corrected sign and the data-driven λ selection; λ is chosen per dataset, not fixed — see `select_lambda.py`):

**Table 2: Normalized Entropy Results**

| Dataset         | K-Means               | ER-K-Means             | Change       |
|-----------------|------------------------|-------------------------|--------------|
| Iris            | 0.9987 ± 0.0009        | 0.9999 ± 0.0000         | +0.12%       |
| Wine            | 0.9952 ± 0.0001        | 0.9973 ± 0.0000         | +0.22%       |
| Breast Cancer   | 0.9211 ± 0.0044        | 0.9753 ± 0.0000         | +5.88%       |
| Digits          | 0.9267 ± 0.0157        | 0.9825 ± 0.0086         | +6.02%       |
| **HTRU2 (real)**| 0.5147 ± 0.0002        | 0.9760 ± 0.0000         | **+89.62%**  |
| syn_3c_60       | 0.8650 ± 0.0000        | 0.9174 ± 0.0084         | +6.06%       |
| syn_3c_70       | 0.7453 ± 0.0000        | 0.9144 ± 0.0005         | +22.69%      |
| syn_3c_80       | 0.5817 ± 0.0000        | 0.8726 ± 0.0186         | +50.01%      |
| syn_3c_90       | 0.3590 ± 0.0000        | 0.8911 ± 0.0383         | +148.21%     |
| **syn_2c_95**   | **0.3221 ± 0.1596**    | **0.9999 ± 0.0000**     | **+210.47%** |

**Table 3: Silhouette Score Results**

| Dataset         | K-Means               | ER-K-Means             | Change       |
|-----------------|------------------------|-------------------------|--------------|
| Iris            | 0.4601 ± 0.0008        | 0.4581 ± 0.0000         | -0.0020      |
| Wine            | 0.2849 ± 0.0003        | 0.2847 ± 0.0000         | -0.0002      |
| Breast Cancer   | 0.3442 ± 0.0008        | 0.3090 ± 0.0000         | -0.0353      |
| Digits          | 0.1447 ± 0.0029        | 0.1240 ± 0.0053         | -0.0208      |
| **HTRU2 (real)**| 0.5993 ± 0.0050        | 0.2604 ± 0.0061         | **-0.3389**  |
| syn_3c_60       | 0.8193 ± 0.0000        | 0.7026 ± 0.0221         | -0.1167      |
| syn_3c_70       | 0.8154 ± 0.0000        | 0.5200 ± 0.0011         | -0.2954      |
| syn_3c_80       | 0.8120 ± 0.0000        | 0.4273 ± 0.0123         | -0.3847      |
| syn_3c_90       | 0.8068 ± 0.0000        | 0.3342 ± 0.0684         | -0.4726      |
| **syn_2c_95**   | **0.6889 ± 0.0717**    | **0.3857 ± 0.0014**     | **-0.3032**  |

Full results, including BKM and GMM-Dirichlet baselines and Adjusted Rand Index against ground truth, are in `results/summary_statistics.csv` and Tables 2–3 of the paper.

## Key Finding (95% Imbalance Case)
- Normalized Entropy: 0.3221 → 0.9999 (+210% improvement)
- Silhouette Score: 0.6889 → 0.3857 (real trade-off — see caveat below)

## Important caveat: balance vs. true structure recovery

**This is new since the original release and matters for how you use this method.** On data where the imbalance reflects genuine underlying structure rather than a synthetic construction, forcing balance can substantially degrade recovery of the true classes. On HTRU2 (real pulsar-detection data, 90.8%/9.2% split), pushing normalized entropy from 0.51 to 0.98 reduces the Adjusted Rand Index against the true pulsar/non-pulsar labels from 0.607 to 0.101. **We do not recommend ER-K-Means for applications like fraud or rare-event detection**, where the minority class's small size is the signal of interest, not an artifact to correct. See Section 6.2 of the paper for the full discussion.


