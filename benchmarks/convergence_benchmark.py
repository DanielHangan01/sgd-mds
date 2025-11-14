from __future__ import annotations
import time
import argparse
from pathlib import Path

import numpy as np
import yaml
from sgd_mds import utils
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import MDS as SklearnMDS
from sklearn.metrics import pairwise_distances as sklearn_pairwise_distances

from sgd_mds.estimator import SGDMDS
from sgd_mds.stress import kruskal_stress_full

MODEL_REGISTRY = {
    "SGDMDS": SGDMDS,
    "SklearnMDS": SklearnMDS,
}


def load_model_specs(config_path: str | Path) -> list[dict]:
    with open(config_path, "r") as fh:
        config = yaml.safe_load(fh) or {}
    models = config.get("models_to_run", [])
    if not models:
        raise ValueError(f"No models found under 'models_to_run' in {config_path}")
    return models


def prepare_model_params(
    model_class: type,
    params: dict,
    *,
    device: torch.device,
    max_iter: int,
) -> dict:
    resolved = dict(params or {})
    resolved["max_iter"] = max_iter
    if model_class is SGDMDS:
        resolved["device"] = device
    elif model_class is SklearnMDS:
        resolved.setdefault("dissimilarity", "precomputed")
        resolved.setdefault("n_init", 1)
    return resolved

def track_convergence(
    model_class: type,
    D_np: np.ndarray,
    max_iters: int,
    model_params: dict,
    device: torch.device,
    use_fair_stress: bool = False,
    stress_weighting: str = utils.PAIR_WEIGHTING_UNIFORM,
    weight_min_delta: float | None = None,
    weight_floor_quantile: float | None = 0.01,
    model_label: str | None = None,
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
    stress_weighting : Weighting used when computing stress (non-uniform always recomputes).
    """
    model_name = model_label or model_class.__name__
    print(f"\n--- Tracking {model_name} Convergence ---")
    
    stress_history = []
    time_history = []
    iter_points = list(range(1, max_iters + 1))
    cumulative_time = 0

    D_t = torch.from_numpy(D_np).to(device)
    weighting_mode = utils.normalize_pair_weighting(stress_weighting)
    min_delta = weight_min_delta
    if (
        min_delta is None
        and weighting_mode != utils.PAIR_WEIGHTING_UNIFORM
        and weight_floor_quantile is not None
    ):
        tri = torch.triu_indices(D_t.size(0), D_t.size(0), offset=1, device=device)
        sampled = D_t[tri[0], tri[1]]
        min_delta = float(torch.quantile(sampled, float(weight_floor_quantile)).item())
    weight_matrix = utils.compute_full_weights(
        D_t,
        weighting_mode,
        min_delta=min_delta,
    )

    for i in iter_points:
        print(f"\rRunning {model_name} for max_iter={i}/{max_iters}", end="")
        
        current_params = model_params.copy()
        current_params['max_iter'] = i
        
        model = model_class(**current_params)
        
        t0 = time.perf_counter()
        X_emb_np = model.fit_transform(D_np)
        t_fit = time.perf_counter() - t0
        cumulative_time += t_fit

        if use_fair_stress or weight_matrix is not None:
            X_emb_t = torch.from_numpy(X_emb_np).to(D_t.device)
            stress = kruskal_stress_full(
                X_emb_t,
                D_t,
                weights=weight_matrix,
            ).item()
        else:
            stress = model.stress_

        stress_history.append(stress)
        time_history.append(cumulative_time)
        
    print("\nDone.")
    return iter_points, stress_history, time_history


def track_sgd_convergence_internal(
    D_np: np.ndarray,
    max_iters: int,
    model_params: dict,
    *,
    model_name: str,
    log_every: int = 1,
) -> tuple[list[int], list[float], list[float]]:
    """
    Runs SGDMDS once while leveraging its built-in convergence logging.
    """
    current_params = model_params.copy()
    current_params["max_iter"] = max_iters
    current_params["track_convergence"] = True
    current_params["convergence_log_every"] = max(1, int(log_every))

    print(
        f"\n--- Running {model_name} (log_every={current_params['convergence_log_every']}) ---"
    )
    model = SGDMDS(**current_params)
    t0 = time.perf_counter()
    model.fit(D_np)
    total_time = time.perf_counter() - t0
    print(f"Completed {model.n_iter_} iterations in {total_time:.2f}s.")

    history = model.convergence_history_
    if not history:
        raise RuntimeError("Convergence tracking is enabled but produced no data.")

    iter_points = [int(entry["iteration"]) for entry in history]
    stress_history = [float(entry["stress"]) for entry in history]
    time_history = [float(entry["elapsed_time"]) for entry in history]
    return iter_points, stress_history, time_history


def main(args: argparse.Namespace) -> None:
    model_specs = load_model_specs(args.config)
    resolved_device = utils.resolve_device(args.device)
    log_every = max(1, args.log_every)

    data_dir = Path(args.dataset_root) / args.dataset
    X_path = data_dir / "X.npy"
    if not X_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {X_path}")
    X_np = np.load(X_path)
    print(f"\nLoaded dataset '{args.dataset}': X={X_np.shape}")

    D_np = sklearn_pairwise_distances(X_np, metric="euclidean").astype(np.float32)
    print(f"Using device: {resolved_device}")

    if args.warmup_runs > 0:
        print("\n--- Performing Warm-up Runs (to stabilize system performance) ---")
        warmup_iter = max(1, min(args.warmup_iter, args.max_iter))
        for warmup_idx in range(args.warmup_runs):
            print(f"Warm-up round {warmup_idx + 1}/{args.warmup_runs}")
            for spec in model_specs:
                model_class = MODEL_REGISTRY.get(spec["class"])
                if model_class is None:
                    raise ValueError(f"Unknown model class '{spec['class']}' in config.")
                warmup_params = prepare_model_params(
                    model_class,
                    spec.get("params", {}),
                    device=resolved_device,
                    max_iter=warmup_iter,
                )
                print(f"  - {spec['name']} (max_iter={warmup_iter})")
                model_instance = model_class(**warmup_params)
                model_instance.fit(D_np)
        print("Warm-up complete.\n")

    convergence_results = []
    for spec in model_specs:
        model_class = MODEL_REGISTRY.get(spec["class"])
        if model_class is None:
            raise ValueError(f"Unknown model class '{spec['class']}' in config.")

        params = prepare_model_params(
            model_class,
            spec.get("params", {}),
            device=resolved_device,
            max_iter=args.max_iter,
        )

        if model_class is SGDMDS:
            iter_points, stress_curve, time_curve = track_sgd_convergence_internal(
                D_np,
                args.max_iter,
                params,
                model_name=spec["name"],
                log_every=log_every,
            )
        else:
            iter_points, stress_curve, time_curve = track_convergence(
                model_class,
                D_np,
                args.max_iter,
                params,
                device=resolved_device,
                use_fair_stress=True,
                stress_weighting=args.stress_weighting,
                weight_floor_quantile=args.stress_weight_floor_quantile,
                model_label=spec["name"],
            )

        convergence_results.append(
            {
                "name": spec["name"],
                "iterations": iter_points,
                "stress": stress_curve,
                "time": time_curve,
            }
        )

    if not convergence_results:
        raise RuntimeError("No convergence results were recorded.")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    for result in convergence_results:
        ax1.plot(
            result["iterations"],
            result["stress"],
            marker=".",
            linestyle="-",
            label=result["name"],
        )
        ax2.plot(
            result["time"],
            result["stress"],
            marker=".",
            linestyle="-",
            label=result["name"],
        )

    ax1.set_xlabel("Number of Iterations")
    ax1.set_ylabel("Normalized Kruskal Stress")
    ax1.set_title("Convergence Speed: Stress vs. Iterations")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_yscale("log")

    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Normalized Kruskal Stress")
    ax2.set_title("Efficiency: Stress vs. Time")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.set_yscale("log")

    fig.suptitle(
        f"MDS Convergence Benchmark on '{args.dataset}' Dataset", fontsize=16
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark MDS convergence behavior using the models defined in benchmark_config.yaml."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="benchmarks/benchmark_config.yaml",
        help="YAML file describing the models to run.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="seismic",
        help="Dataset subdirectory name under --dataset_root.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="datasets",
        help="Root directory containing dataset folders.",
    )
    parser.add_argument("--max_iter", type=int, default=100, help="Max iterations to track.")
    parser.add_argument(
        "--log_every",
        type=int,
        default=1,
        help="Record SGDMDS stress every N iterations when using the internal tracker.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device for SGDMDS.")
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=5,
        help="Number of untimed runs per model before benchmarking.",
    )
    parser.add_argument(
        "--warmup_iter",
        type=int,
        default=10,
        help="Iteration cap used during each warm-up run.",
    )
    parser.add_argument(
        "--stress_weighting",
        type=str,
        default=utils.PAIR_WEIGHTING_CHOICES[0],
        choices=utils.PAIR_WEIGHTING_CHOICES,
        help="Weighting scheme applied when plotting stress curves for non-SGDMDS models.",
    )
    parser.add_argument(
        "--stress_weight_floor_quantile",
        type=float,
        default=0.01,
        help="Quantile used to clamp inverse-distance weights in stress calculations.",
    )
    args = parser.parse_args()
    main(args)
