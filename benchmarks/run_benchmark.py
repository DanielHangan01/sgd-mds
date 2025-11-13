from __future__ import annotations
import argparse
import time
import yaml
import numpy as np
from sklearn.manifold import MDS as SklearnMDS
from sklearn.metrics import pairwise_distances as sklearn_pairwise_distances

from sgd_mds.estimator import SGDMDS
from sgd_mds.utils import (
    PAIR_WEIGHTING_CHOICES,
    resolve_device,
)

from benchmark_utils import (
    load_dataset,
    calculate_fair_stress,
    print_summary,
    plot_results,
    BenchmarkResult,
)

MODEL_REGISTRY = {
    "SGDMDS": SGDMDS,
    "SklearnMDS": SklearnMDS,
}

def main(args: argparse.Namespace) -> None:
    # --- Load Configuration and Data ---
    print(f"--- Loading benchmark configuration from: {args.config} ---")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    models_to_run_config = config.get("models_to_run", [])
    
    X, y = load_dataset(args.dataset)
    if args.n_samples:
        X, y = X[:args.n_samples], y[:args.n_samples]

    print("\n[Step 1] Computing full pairwise distance matrix...")
    D_np = sklearn_pairwise_distances(X, metric="euclidean").astype(np.float32)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # --- Run Benchmark for Each Model ---
    final_results = []
    for model_config in models_to_run_config:
        model_name = model_config["name"]
        model_class = MODEL_REGISTRY[model_config["class"]]
        params = model_config.get("params", {}).copy()
        
        print("\n" + "="*50)
        print(f"Benchmarking Model: {model_name}")
        print("="*50)

        params['max_iter'] = args.max_iter
        if model_class is SGDMDS:
            params['device'] = device

        # --- WARM-UP (BURN-IN) PHASE ---
        if args.n_warmup > 0:
            print(f"Performing {args.n_warmup} warm-up runs...")
            for _ in range(args.n_warmup):
                model_instance = model_class(**params)
                model_instance.fit(D_np)
        
        # --- TIMED MEASUREMENT PHASE ---
        trial_results = []
        print(f"Performing {args.n_trials} timed trial(s)...")
        for i in range(args.n_trials):
            model_instance = model_class(**params)
            
            t0 = time.perf_counter()
            embedding = model_instance.fit_transform(D_np)
            fit_time = time.perf_counter() - t0
            
            stress_device = device if model_class is SGDMDS else "cpu"
            stress = calculate_fair_stress(
                embedding,
                D_np,
                stress_device,
                weighting=args.stress_weighting,
                weight_min_delta=getattr(model_instance, "pair_weight_min_delta_", None),
                weight_floor_quantile=args.stress_weight_floor_quantile,
            )
            
            n_iter = getattr(model_instance, 'n_iter_', 0)
            
            result = BenchmarkResult(model_name, embedding, fit_time, stress, n_iter)
            trial_results.append(result)
            print(f"  Trial {i+1}/{args.n_trials}: Time={fit_time:.4f}s, Stress={stress:.6f}")

        # --- AGGREGATE RESULTS FOR THIS MODEL ---
        avg_fit_time = np.mean([res.fit_time for res in trial_results])
        avg_stress = np.mean([res.stress for res in trial_results])
        avg_n_iter = int(np.mean([res.n_iter for res in trial_results]))
        
        # Use the embedding from the last trial for plotting
        last_embedding = trial_results[-1].embedding
        
        final_results.append(
            BenchmarkResult(model_name, last_embedding, avg_fit_time, avg_stress, avg_n_iter)
        )

    # --- Report and Visualize Final Averaged Results ---
    print("\n" + "="*60)
    print(" " * 15 + "Final Averaged Benchmark Results")
    print("="*60)
    print_summary(final_results)
    plot_results(final_results, y, args.dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a robust, multi-trial comparative benchmark for MDS algorithms with burn-in."
    )
    parser.add_argument("dataset", type=str, help="Name of the dataset folder.")
    parser.add_argument("--config", type=str, default="benchmarks/benchmark_config.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_iter", type=int, default=10_000)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--n_warmup", type=int, default=2, help="Number of untimed burn-in runs per model.")
    parser.add_argument("--n_trials", type=int, default=5, help="Number of timed trials to average per model.")
    parser.add_argument(
        "--stress_weighting",
        type=str,
        default=PAIR_WEIGHTING_CHOICES[0],
        choices=PAIR_WEIGHTING_CHOICES,
        help="Weighting applied when reporting benchmark stress metrics.",
    )
    parser.add_argument(
        "--stress_weight_floor_quantile",
        type=float,
        default=0.01,
        help="Quantile used to clamp inverse-distance stress calculations.",
    )
    args = parser.parse_args()
    main(args)
