from __future__ import annotations
import time
import argparse
from pathlib import Path

import numpy as np
from sgd_mds import utils
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import MDS as SklearnMDS
from sklearn.metrics import pairwise_distances as sklearn_pairwise_distances

from sgd_mds.estimator import SGDMDS
from sgd_mds.stress import kruskal_stress_full

def track_convergence(
    model_class: type,
    D_np: np.ndarray,
    max_iters: int,
    model_params: dict,
    device: torch.device,
    use_fair_stress: bool = False,
) -> tuple[list[int], list[float], list[float]]:
    """
    Tracks model convergence by repeatedly fitting with increasing max_iter.

    This "black-box" method treats any MDS model the same way, providing a
    fair comparison by re-running the fit process for each iteration count.

    Parameters
    ----------
    model_class : The MDS model class to benchmark (e.g., SGDMDS).
    D_np : The precomputed distance matrix.
    max_iters : The maximum number of iterations to track.
    model_params : A dictionary of parameters to initialize the model.
    use_fair_stress : If True, recalculates stress using our normalized formula.
    """
    model_name = model_class.__name__
    print(f"\n--- Tracking {model_name} Convergence ---")
    
    stress_history = []
    time_history = []
    iter_points = list(range(1, max_iters + 1))
    cumulative_time = 0

    D_t = torch.from_numpy(D_np).to(device)

    for i in iter_points:
        print(f"\rRunning {model_name} for max_iter={i}/{max_iters}", end="")
        
        current_params = model_params.copy()
        current_params['max_iter'] = i
        
        model = model_class(**current_params)
        
        t0 = time.perf_counter()
        X_emb_np = model.fit_transform(D_np)
        t_fit = time.perf_counter() - t0
        cumulative_time += t_fit

        if use_fair_stress:
            X_emb_t = torch.from_numpy(X_emb_np).to(D_t.device)
            stress = kruskal_stress_full(X_emb_t, D_t).item()
        else:
            stress = model.stress_

        stress_history.append(stress)
        time_history.append(t_fit)
        
    print("\nDone.")
    return iter_points, stress_history, time_history


def main(args: argparse.Namespace) -> None:
    # 1. --- Data Loading & Preprocessing ---
    data_dir = Path("datasets/seismic")
    X_np = np.load(data_dir / "X.npy")
    n = len(X_np)
    print(f"\nLoaded seismic dataset: X={X_np.shape}")
    
    D_np = sklearn_pairwise_distances(X_np, metric="euclidean").astype(np.float32)

    resolved_device = utils.resolve_device(args.device)
    print(f"Using device: {resolved_device}")
    
    sgd_mds_params = {
        "n_components": 2,
        "stopper": "iterations",
        "lr_init": 0.01,
        "scheduler": "constant",
        "random_state": 42,
        "device": resolved_device,
    }

    sklearn_mds_params = {
        "n_components": 2,
        "dissimilarity": "precomputed",
        "n_init": 1,
        "random_state": 42,
        "n_jobs": -1,
    }

    print("\n--- Performing Warm-up Runs (to stabilize system performance) ---")
    for i in range(args.warmup_runs):
        print(f"\rWarm-up run {i+1}/{args.warmup_runs}", end="")
        SklearnMDS(**sklearn_mds_params, max_iter=10).fit(D_np)
        SGDMDS(**sgd_mds_params, max_iter=10).fit(D_np)
    print("\nWarm-up complete.")

    # Run Convergence Benchmarks
    sgd_iters, sgd_stress, sgd_time = track_convergence(
        SGDMDS, D_np, args.max_iter, sgd_mds_params,
        device=resolved_device, 
        use_fair_stress=False
    )
    sk_iters, sk_stress, sk_time = track_convergence(
        SklearnMDS, D_np, args.max_iter, sklearn_mds_params, 
        device=resolved_device, 
        use_fair_stress=True
    )

    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    
    # Plot 1: Stress vs. Iteration
    ax1.plot(sk_iters, sk_stress, marker='.', linestyle='-', label="Scikit-learn MDS (SMACOF)")
    ax1.plot(sgd_iters, sgd_stress, marker='.', linestyle='-', label="SGD-MDS")
    ax1.set_xlabel("Number of Iterations")
    ax1.set_ylabel("Normalized Kruskal Stress")
    ax1.set_title("Convergence Speed: Stress vs. Iterations")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_yscale('log')

    # Plot 2: Stress vs. Time
    ax2.plot(sk_time, sk_stress, marker='.', linestyle='-', label="Scikit-learn MDS (SMACOF)")
    ax2.plot(sgd_time, sgd_stress, marker='.', linestyle='-', label="SGD-MDS")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Normalized Kruskal Stress")
    ax2.set_title("Efficiency: Stress vs. Time")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_yscale('log')

    fig.suptitle("MDS Convergence Benchmark on Seismic Dataset (Black-Box Method)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark MDS convergence behavior using a 'black-box' repeated-fit method."
    )
    parser.add_argument("--max_iter", type=int, default=100, help="Max iterations to track.")
    parser.add_argument("--device", type=str, default="auto", help="Device for SGDMDS.")
    parser.add_argument("--warmup_runs", type=int, default=5, help="Number of untimed runs to perform before benchmarking.")
    args = parser.parse_args()
    main(args)