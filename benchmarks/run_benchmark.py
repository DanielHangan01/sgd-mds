from __future__ import annotations
import argparse
import yaml
import numpy as np
from sklearn.manifold import MDS as SklearnMDS
from sklearn.metrics import pairwise_distances as sklearn_pairwise_distances

from sgd_mds.estimator import SGDMDS
from sgd_mds.utils import resolve_device

from benchmark_utils import (
    load_dataset,
    fit_and_time_model,
    calculate_fair_stress,
    print_summary,
    plot_results,
    BenchmarkResult,
)

# A registry to map class names from the YAML file to actual Python classes
MODEL_REGISTRY = {
    "SGDMDS": SGDMDS,
    "SklearnMDS": SklearnMDS,
}

def main(args: argparse.Namespace) -> None:
    # Load Benchmark Configuration
    print(f"--- Loading benchmark configuration from: {args.config} ---")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    models_to_run_config = config.get("models_to_run", [])
    if not models_to_run_config:
        raise ValueError("Configuration file must contain a 'models_to_run' list.")

    # Load Dataset
    X, y = load_dataset(args.dataset)
    if args.n_samples is not None and args.n_samples < len(X):
        print(f"Subsampling to {args.n_samples} points.")
        X, y = X[:args.n_samples], y[:args.n_samples]

    # Preprocessing
    print("\n[Step 1] Computing full pairwise distance matrix...")
    D_np = sklearn_pairwise_distances(X, metric="euclidean").astype(np.float32)

    # Run Benchmarks
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    
    results = []
    for model_config in models_to_run_config:
        model_name = model_config["name"]
        model_class_name = model_config["class"]
        params = model_config.get("params", {})

        print(f"\n--- Configuring Model: {model_name} ---")
        
        if model_class_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model class '{model_class_name}' in config.")
        model_class = MODEL_REGISTRY[model_class_name]

        params['max_iter'] = args.max_iter
        if model_class is SGDMDS:
            params['device'] = device

        model = model_class(**params)
        
        embedding, fit_time = fit_and_time_model(model, D_np)
        
        stress_device = device if model_class is SGDMDS else "cpu"
        stress = calculate_fair_stress(embedding, D_np, stress_device)
        
        n_iter = getattr(model, 'n_iter_', 0)
        
        results.append(BenchmarkResult(model_name, embedding, fit_time, stress, n_iter))

    # Visualization & Summary
    print_summary(results)
    plot_results(results, y, args.dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a comparative benchmark for MDS algorithms using a YAML configuration file."
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Name of the dataset folder in 'datasets/' (e.g., 'seismic')."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="benchmarks/benchmark_config.yaml",
        help="Path to the YAML file defining the models to benchmark."
    )
    parser.add_argument("--device", type=str, default="auto", help="Device for SGD-MDS.")
    parser.add_argument("--max_iter", type=int, default=300, help="Max iterations for the models.")
    parser.add_argument("--n_samples", type=int, default=None, help="Subsample dataset points.")
    args = parser.parse_args()
    main(args)