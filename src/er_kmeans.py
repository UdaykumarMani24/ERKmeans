#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ER-K-Means: Entropy-Regularized K-Means Clustering
Version 2.1 - DEBUGGED - Fixed assignment issues
"""

import numpy as np
from sklearn.metrics import pairwise_distances
import warnings

warnings.filterwarnings('ignore')


class ERKMeans:
    """
    Entropy-Regularized K-Means Clustering
    """
    
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
    
    def fit(self, X):
        """Fit ER-K-Means to data X"""
        X = np.asarray(X)
        n = X.shape[0]
        
        # Initialize centroids using K-Means++
        self.centroids_ = self._kmeans_plus_plus_init(X)
        
        # Initialize cluster sizes to 1 (avoid log(0))
        self.cluster_sizes_ = np.ones(self.k, dtype=int)
        self.labels_ = np.zeros(X.shape[0], dtype=int)
        self.n_iter_ = 0
        self.objective_history_ = []
        
        # Initial assignment
        self.labels_ = self._assign_points_batch(X, n)
        self._update_sizes_and_centroids(X)
        
        for iteration in range(self.max_iters):
            self.n_iter_ = iteration + 1
            
            # Store previous
            prev_centroids = self.centroids_.copy()
            prev_labels = self.labels_.copy()
            
            # Assignment step
            new_labels = self._assign_points_batch(X, n)
            
            # Update centroids based on new labels
            new_centroids = np.zeros_like(self.centroids_)
            new_sizes = np.zeros(self.k, dtype=int)
            
            for i in range(self.k):
                mask = new_labels == i
                new_sizes[i] = np.sum(mask)
                if new_sizes[i] > 0:
                    new_centroids[i] = X[mask].mean(axis=0)
                else:
                    # Empty cluster: reinitialize
                    distances = pairwise_distances(X, self.centroids_).min(axis=1)
                    farthest_idx = np.argmax(distances)
                    new_centroids[i] = X[farthest_idx]
                    new_sizes[i] = 1  # Prevent division by zero
            
            # Compute objective
            objective = self._compute_objective(X, n, new_labels, new_sizes, new_centroids)
            self.objective_history_.append(objective)
            
            # Update
            self.labels_ = new_labels
            self.centroids_ = new_centroids
            self.cluster_sizes_ = new_sizes
            
            # Check convergence
            centroid_shift = np.linalg.norm(self.centroids_ - prev_centroids)
            labels_changed = not np.array_equal(self.labels_, prev_labels)
            
            if self.verbose:
                print(f"Iter {self.n_iter_}: Obj={objective:.2f}, Shift={centroid_shift:.6f}, "
                      f"Sizes={self.cluster_sizes_}")
            
            if centroid_shift < self.tol and not labels_changed:
                if self.verbose:
                    print(f"Converged at iteration {self.n_iter_}")
                break
        
        # Final update
        self._update_sizes_and_centroids(X)
        
        return self
    
    def _update_sizes_and_centroids(self, X):
        """Update cluster sizes and centroids from current labels"""
        for i in range(self.k):
            mask = self.labels_ == i
            self.cluster_sizes_[i] = np.sum(mask)
            if self.cluster_sizes_[i] > 0:
                self.centroids_[i] = X[mask].mean(axis=0)
    
    def _kmeans_plus_plus_init(self, X):
        """K-Means++ initialization"""
        n = X.shape[0]
        centroids = []
        
        # First centroid: random point
        first_idx = np.random.choice(n)
        centroids.append(X[first_idx])
        
        # Select remaining centroids
        for _ in range(1, self.k):
            distances = pairwise_distances(X, np.array(centroids)).min(axis=1)
            if distances.sum() == 0:
                distances = np.ones(n)
            probabilities = distances / distances.sum()
            next_idx = np.random.choice(n, p=probabilities)
            centroids.append(X[next_idx])
        
        return np.array(centroids)
    
    def _assign_points_batch(self, X, n):
        """
        Batch assignment with entropy regularization.
        CRITICAL FIX: Normalize lambda appropriately.
        """
        # Compute distances to all centroids
        distances = pairwise_distances(X, self.centroids_)
        
        # Apply entropy correction
        # Formula: score = dist - (lambda / n) * log(size / n)
        # But we need to ensure numerical stability
        for i in range(self.k):
            if self.cluster_sizes_[i] > 0:
                size_norm = self.cluster_sizes_[i] / n
                # Clamp to avoid log(very small) -> -inf
                size_norm = max(size_norm, 1e-10)
                # Calculate correction
                correction = (self.lambda_reg / n) * np.log(size_norm)
                distances[:, i] = distances[:, i] - correction
            else:
                # Empty cluster: make it slightly attractive to fill
                distances[:, i] = distances[:, i] - 1e-6
        
        # Assign to minimum distance
        labels = np.argmin(distances, axis=1)
        
        # Debug: print cluster sizes after assignment
        if self.verbose:
            sizes = np.bincount(labels, minlength=self.k)
            print(f"    Assignment result: sizes={sizes}")
        
        return labels
    
    def _compute_objective(self, X, n, labels, cluster_sizes, centroids):
        """Compute ER-K-Means objective"""
        # Distortion
        distortion = 0.0
        for i in range(self.k):
            if cluster_sizes[i] > 0:
                cluster_points = X[labels == i]
                distortion += np.sum((cluster_points - centroids[i]) ** 2)
        
        # Entropy (using all k clusters)
        p = np.maximum(cluster_sizes / n, 1e-12)  # Avoid log(0)
        entropy = -np.sum(p * np.log(p))
        
        return distortion - self.lambda_reg * entropy
    
    def predict(self, X):
        """Predict labels for new data"""
        X = np.asarray(X)
        n = X.shape[0]
        
        distances = pairwise_distances(X, self.centroids_)
        
        for i in range(self.k):
            if self.cluster_sizes_[i] > 0:
                size_norm = max(self.cluster_sizes_[i] / n, 1e-10)
                correction = (self.lambda_reg / n) * np.log(size_norm)
                distances[:, i] = distances[:, i] - correction
        
        return np.argmin(distances, axis=1)
    
    def get_balance_metrics(self):
        """Get balance metrics"""
        n = sum(self.cluster_sizes_)
        k = self.k
        p = self.cluster_sizes_ / n
        entropy = -np.sum(p * np.log(p + 1e-12))
        h_norm = entropy / np.log(k) if np.log(k) > 0 else 0
        
        return {
            'h_norm': h_norm,
            'entropy': entropy,
            'sizes': self.cluster_sizes_.tolist()
        }