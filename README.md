# Entropy-Regularized K-Means (ER-K-Means)

This repository contains the implementation and experiments for the paper:  
**"Entropy-Regularized K-Means: A Provably Convergent Framework for Preventing Cluster Collapse"**

## Authors

- Udayakumar Mani (uthay@bioinfo.sastra.edu)
- Lavanya Priyadarshini Ramalingam
- Senthilkumar Rathinasamy (senthilrathna@sastra.ac.in)

**Green Separation Engineering Laboratory, SASTRA Deemed to be University, Thanjavur, Tamil Nadu, 613 401, India**

---
<pre>
## Repository Structure
├── data/ # Dataset files (Iris, Wine, Heart Disease, synthetic)
├── figures/ # Generated figures
│ ├── figure1_balance_comparison.png
│ └── figure2_runtime_comparison.png
├── results/ # Experiment output files
│ ├── experiment_results.csv
│ ├── real_experiment_results.csv
│ └── summary_statistics.csv
├── src/ # Source code
│ ├── pycache/ # Compiled Python files (ignored by git)
│ ├── analysis.py # Analysis utilities
│ ├── analyze_results.py # Results analysis script
│ ├── data_loader.py # Dataset loading functions
│ ├── er_kmeans.py # Main ER-K-Means implementation
│ ├── experiments.py # Experiment runner
│ ├── f1.py # Additional experiments
│ ├── run_experiments.py # Main entry point
│ └── results/ # Intermediate results
│ ├── experiment_results.csv
│ └── summary_statistics.csv
└── README.md # This file

</pre>


---

## Requirements

- Python 3.8 or higher
- Required packages:

```bash
pip install numpy scikit-learn scipy matplotlib pandas

## 1.Running Experiments
cd src
python run_experiments.py

## 2. Run individual dataset experiments
cd src

# Real datasets (balanced)
python experiments.py --dataset iris --runs 30
python experiments.py --dataset wine --runs 30
python experiments.py --dataset heart --runs 30

# Synthetic datasets (imbalanced)
python experiments.py --dataset syn_3c_60 --runs 30   # 60% imbalance
python experiments.py --dataset syn_3c_70 --runs 30   # 70% imbalance
python experiments.py --dataset syn_3c_80 --runs 30   # 80% imbalance
python experiments.py --dataset syn_3c_90 --runs 30   # 90% imbalance
python experiments.py --dataset syn_2c_95 --runs 30   # 95% imbalance (extreme)




Expected Results
After running the experiments, you should obtain results matching the paper:

Table 2: Normalized Entropy Results (mean ± std over 30 runs)

| Dataset         | K-Means               | ER-K-Means            | Change       |
|-----------------|-----------------------|-----------------------|--------------|
| Iris            | 0.9986 ± 0.0009       | 0.9691 ± 0.0732       | -2.96%       |
| Wine            | 0.9952 ± 0.0001       | 0.9940 ± 0.0014       | -0.13%       |
| Heart Disease   | 0.9563 ± 0.0007       | 0.9612 ± 0.0106       | +0.51%       |
| syn_3c_60       | 0.8650 ± 0.0000       | 0.8555 ± 0.0512       | -1.09%       |
| syn_3c_70       | 0.7453 ± 0.0000       | 0.7275 ± 0.0669       | -2.38%       |
| syn_3c_80       | 0.5817 ± 0.0000       | 0.5943 ± 0.0683       | +2.17%       |
| syn_3c_90       | 0.3590 ± 0.0000       | 0.3758 ± 0.0911       | +4.69%       |
| **syn_2c_95**   | **0.3102 ± 0.1303**   | **0.8570 ± 0.2869**   | **+176.28%** |


Table 3: Silhouette Score Results (mean ± std over 30 runs)

| Dataset         | K-Means               | ER-K-Means            | Change       |
|-----------------|-----------------------|-----------------------|--------------|
| Iris            | 0.4592 ± 0.0010       | 0.4628 ± 0.0046       | +0.0036      |
| Wine            | 0.2849 ± 0.0000       | 0.2830 ± 0.0025       | -0.0019      |
| Heart Disease   | 0.1719 ± 0.0000       | 0.1710 ± 0.0023       | -0.0009      |
| syn_3c_60       | 0.8193 ± 0.0000       | 0.8080 ± 0.0097       | -0.0113      |
| syn_3c_70       | 0.8154 ± 0.0000       | 0.7976 ± 0.0138       | -0.0178      |
| syn_3c_80       | 0.8120 ± 0.0000       | 0.7969 ± 0.0173       | -0.0151      |
| syn_3c_90       | 0.8068 ± 0.0000       | 0.7927 ± 0.0221       | -0.0141      |
| **syn_2c_95**   | **0.6943 ± 0.0420**   | **0.4486 ± 0.1087**   | **-0.2457**  |


Key Finding (95% Imbalance Case)
Normalized Entropy: 0.3102 → 0.8570 (176% improvement)

Silhouette Score: 0.6943 → 0.4486 (moderate trade-off)

Effect Size (Cohen's d): 1.21 (large effect)

</pre>
