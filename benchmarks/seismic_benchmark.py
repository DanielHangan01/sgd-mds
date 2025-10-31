from __future__ import annotations
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import MDS as SklearnMDS
from sklearn.metrics import pairwise_distances as sklearn_pairwise_distances

from sgd_mds.estimator import SGDMDS
from sgd_mds.stress import kruskal_stress_full

def main(args: argparse.Namespace) -> None:
    print("--- SGD-MDS vs. Scikit-learn MDS Benchmark ---")

    data_dir = Path("datasets/seismic")
    try:
        X = np.load(data_dir / "X.npy")
        y = np.load(data_dir / "y.npy")
    except FileNotFoundError:
        print("\nERROR: Dataset not found.")
        return

    n = len(X)
    print(f"\nLoaded seismic dataset: X={X.shape}, y={y.shape}")

    print("\n[Step 1] Computing full pairwise distance matrix...")
    t0 = time.perf_counter()
    D_np = sklearn_pairwise_distances(X, metric="euclidean").astype(np.float32)
    t_dist = time.perf_counter() - t0
    print(f"Done in {t_dist:.2f} seconds.")

    print("\n[Step 2] Running SGDMDS...")
    model_sgd = SGDMDS(
        n_components=2,
        stopper="iterations",
        max_iter=239,
        lr_init=0.01,
        scheduler="constant",
        random_state=42,
        device=args.device,
    )
    t0 = time.perf_counter()
    X_emb_sgd_np = model_sgd.fit_transform(D_np)
    t_sgd = time.perf_counter() - t0

    print("\n[Step 3] Running Scikit-learn's MDS (SMACOF)...")
    model_sklearn = SklearnMDS(
        n_components=2,
        dissimilarity="precomputed",
        n_init=1,
        max_iter=300,
        random_state=42,
        n_jobs=-1,
    )
    t0 = time.perf_counter()
    X_emb_sklearn_np = model_sklearn.fit_transform(D_np)
    t_sklearn = time.perf_counter() - t0

    device = torch.device(args.device if args.device != "auto" else "cpu")
    D_t = torch.from_numpy(D_np).to(device)
    X_emb_sgd_t = torch.from_numpy(X_emb_sgd_np).to(device)
    X_emb_sklearn_t = torch.from_numpy(X_emb_sklearn_np).to(device)

    stress_sgd = kruskal_stress_full(X_emb_sgd_t, D_t).item()
    stress_sklearn = kruskal_stress_full(X_emb_sklearn_t, D_t).item()
    
    print("\n--- Benchmark Summary ---")
    print(f"{'Metric':<25} | {'SGD-MDS':<20} | {'Scikit-learn MDS':<20}")
    print("-" * 70)
    print(f"{'Fit Time (s)':<25} | {t_sgd:<20.4f} | {t_sklearn:<20.4f}")
    print(f"{'Normalized Kruskal Stress':<25} | {stress_sgd:<20.6f} | {stress_sklearn:<20.6f}")
    print(f"{'Iterations':<25} | {model_sgd.n_iter_:<20} | {model_sklearn.n_iter_:<20}")
    print("-" * 70)
    print(f"(Scikit-learn's raw stress was: {model_sklearn.stress_:.2f})")


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    ax1.scatter(X_emb_sgd_np[:, 0], X_emb_sgd_np[:, 1], c=y, cmap="coolwarm", s=25, alpha=0.9)
    ax1.set_title(f"SGD-MDS\n(Stress={stress_sgd:.4f})")
    ax1.set_xlabel("Dimension 1")
    ax1.set_ylabel("Dimension 2")
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.scatter(X_emb_sklearn_np[:, 0], X_emb_sklearn_np[:, 1], c=y, cmap="coolwarm", s=25, alpha=0.9)
    ax2.set_title(f"Scikit-learn MDS (SMACOF)\n(Stress={stress_sklearn:.4f})")
    ax2.set_xlabel("Dimension 1")
    ax2.set_ylabel("Dimension 2")
    ax2.grid(True, linestyle='--', alpha=0.6)

    fig.suptitle(f"MDS Embedding of the Seismic Dataset (n={n})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark SGD-MDS against Scikit-learn's MDS on the Seismic dataset."
    )
    parser.add_argument("--device", type=str, default="auto", help="Device to run.")
    args = parser.parse_args()
    main(args)