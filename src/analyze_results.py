#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Statistical Analysis and Plot Generation for ER-K-Means Results
Version 2.0 - Production Ready for Manuscript Publication
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib/seaborn not installed. Install with: pip install matplotlib seaborn")
    PLOTTING_AVAILABLE = False


def load_results(results_path='results/experiment_results.csv'):
    """Load experiment results"""
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results not found at {results_path}. Run run_experiments.py first.")
    
    df = pd.read_csv(results_path)
    print(f"Loaded {len(df)} experiments from {results_path}")
    return df


def perform_statistical_tests(df):
    """
    Perform statistical analysis comparing K-Means vs ER-K-Means
    
    Returns
    -------
    results_df : pd.DataFrame
        Statistical test results
    """
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)
    
    statistical_results = []
    
    for dataset in df['dataset'].unique():
        print(f"\n{dataset.upper()}:")
        
        # Get K-Means data
        km_data = df[(df['dataset'] == dataset) & (df['method'] == 'K-Means')]['h_norm'].values
        
        if len(km_data) == 0:
            print(f"  No K-Means data for {dataset}")
            continue
        
        # Find best ER-K-Means lambda for this dataset
        erk_by_lambda = {}
        for lambda_val in df[df['method'] == 'ER-K-Means']['lambda'].unique():
            erk_data = df[(df['dataset'] == dataset) & 
                         (df['method'] == 'ER-K-Means') & 
                         (df['lambda'] == lambda_val)]['h_norm'].values
            if len(erk_data) > 0:
                erk_by_lambda[lambda_val] = erk_data
        
        if not erk_by_lambda:
            print(f"  No ER-K-Means data for {dataset}")
            continue
        
        # Find lambda with best balance
        best_lambda = max(erk_by_lambda.keys(), 
                         key=lambda l: erk_by_lambda[l].mean())
        erk_best = erk_by_lambda[best_lambda]
        
        # Also get GMM data if available
        gmm_data = df[(df['dataset'] == dataset) & (df['method'] == 'GMM-Dirichlet')]['h_norm'].values
        
        # Wilcoxon signed-rank test
        if len(km_data) == len(erk_best):
            stat, p_value = stats.wilcoxon(km_data, erk_best)
            
            # Calculate improvement
            improvement = (erk_best.mean() - km_data.mean()) / km_data.mean() * 100
            
            # Cohen's d effect size
            pooled_std = np.sqrt((np.var(km_data) + np.var(erk_best)) / 2)
            cohens_d = (erk_best.mean() - km_data.mean()) / pooled_std if pooled_std > 0 else 0
            
            # Effect size interpretation
            if abs(cohens_d) < 0.2:
                effect = "negligible"
            elif abs(cohens_d) < 0.5:
                effect = "small"
            elif abs(cohens_d) < 0.8:
                effect = "medium"
            else:
                effect = "large"
            
            #