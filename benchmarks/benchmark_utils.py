from __future__ import annotations
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances as sklearn_pairwise_distances

from sgd_mds.stress import kruskal_stress_full

class BenchmarkResult:
    """A simple data class to hold the results of a single benchmark run."""
    def __init__(self, name: str, embedding: np.ndarray, fit_time: float,
                 stress: float, n_iter: int):
        self.name = name
        self.embedding = embedding
        self.fit_time = fit_time
        self.stress = stress
        self.n_iter = n_iter

def load_dataset(name: str, path_prefix: str = "datasets") -> Tuple[np.ndarray, np.ndarray]:
    """Loads a dataset (X.npy, y.npy) from a subdirectory."""
    data_dir = Path(path_prefix) / name
    print(f"--- Loading dataset: '{name}' ---")
    try:
        X = np.load(data_dir / "X.npy")
        y = np.load(data_dir / "y.npy")
        print(f"Loaded successfully: X={X.shape}, y={y.shape}")
        return X, y
    except FileNotFoundError:
        print(f"\nERROR: Dataset '{name}' not found in '{data_dir}/'")
        print("Please ensure the directory exists and contains X.npy and y.npy files.")
        raise

def fit_and_time_model(model: Any, D_np: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fits an MDS model and returns the embedding and wall-clock time."""
    print(f"Running {model.__class__.__name__}...")
    t0 = time.perf_counter()
    embedding = model.fit_transform(D_np)
    fit_time = time.perf_counter() - t0
    return embedding, fit_time

def calculate_fair_stress(embedding_np: np.ndarray, D_np: np.ndarray, device: torch.device) -> float:
    """Calculates the normalized Kruskal stress for any given embedding."""
    D_t = torch.from_numpy(D_np).to(device)
    embedding_t = torch.from_numpy(embedding_np).to(device)
    stress = kruskal_stress_full(embedding_t, D_t).item()
    return stress

def print_summary(results: List[BenchmarkResult]):
    """Prints a formatted summary table of the benchmark results."""
    print("\n--- Benchmark Summary ---")
    header = f"{'Metric':<25} |"
    for res in results:
        header += f" {res.name:<20} |"
    print(header)
    print("-" * len(header))
    
    fit_time_row = f"{'Fit Time (s)':<25} |"
    stress_row = f"{'Normalized Kruskal Stress':<25} |"
    iters_row = f"{'Iterations':<25} |"
    
    for res in results:
        fit_time_row += f" {res.fit_time:<20.4f} |"
        stress_row += f" {res.stress:<20.6f} |"
        iters_row += f" {res.n_iter:<20} |"
        
    print(fit_time_row)
    print(stress_row)
    print(iters_row)
    print("-" * len(header))

def plot_results(results: List[BenchmarkResult], y: np.ndarray, dataset_name: str):
    """Plots the embeddings from all benchmark runs side-by-side."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5.5), 
                             sharex=True, sharey=True)
    if n_models == 1:
        axes = [axes]

    for i, res in enumerate(results):
        ax = axes[i]
        ax.scatter(res.embedding[:, 0], res.embedding[:, 1], c=y, cmap="coolwarm", 
                   s=25, alpha=0.9)
        ax.set_title(f"{res.name}\n(Stress={res.stress:.4f})")
        ax.set_xlabel("Dimension 1")
        if i == 0:
            ax.set_ylabel("Dimension 2")
        ax.grid(True, linestyle='--', alpha=0.6)

    fig.suptitle(f"MDS Embedding of the '{dataset_name.capitalize()}' Dataset (n={len(y)})", 
                 fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()