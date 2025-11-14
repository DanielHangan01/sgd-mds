from __future__ import annotations

import argparse
import copy
import itertools
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import yaml
from sklearn.metrics import pairwise_distances as sklearn_pairwise_distances

from sgd_mds.estimator import SGDMDS
from sgd_mds.utils import resolve_device
from benchmark_utils import (
    load_dataset,
    calculate_fair_stress,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid-search tuner for SGDMDS stoppers/schedulers/hyperparameters."
    )
    parser.add_argument("dataset", type=str, help="Dataset name (subdirectory under datasets/).")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("benchmarks/hparam_search.yaml"),
        help="YAML file describing experiments and search grids.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device for SGDMDS.")
    parser.add_argument("--stress_device", type=str, default="cpu", help="Device used to compute stress.")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Optional subsample of the dataset (applied after loading).",
    )
    parser.add_argument(
        "--max_configs",
        type=int,
        default=None,
        help="Maximum number of configurations evaluated per experiment.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="How many best configs to display per experiment.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=None,
        help="Override the model's max_iter for every configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save raw search results as JSON.",
    )
    return parser.parse_args()


def load_search_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Search configuration file not found: {path}")
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if not data or "experiments" not in data:
        raise ValueError("Search config must contain an 'experiments' list.")
    return data


def iter_search_points(search_spec: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
    if not search_spec:
        yield {}
        return
    keys = list(search_spec.keys())
    values_product = itertools.product(*(search_spec[k] for k in keys))
    for combo in values_product:
        yield dict(zip(keys, combo))


def set_nested_value(params: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = params
    for key in parts[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[parts[-1]] = value


def evaluate_config(
    model_name: str,
    params: Dict[str, Any],
    D_np: np.ndarray,
    stress_device: str,
) -> Dict[str, Any]:
    start = time.perf_counter()
    model = SGDMDS(**params)
    model.fit(D_np)
    fit_time = time.perf_counter() - start

    stress = calculate_fair_stress(
        model.embedding_,
        D_np,
        resolve_device(stress_device),
        weighting=params.get("pair_weighting", "uniform"),
        weight_min_delta=getattr(model, "pair_weight_min_delta_", None),
    )

    return {
        "model_name": model_name,
        "params": params,
        "stress": stress,
        "fit_time": fit_time,
        "n_iter": model.n_iter_,
    }


def main() -> None:
    args = parse_args()
    config = load_search_config(args.config)

    X, y = load_dataset(args.dataset)
    if args.n_samples:
        X = X[: args.n_samples]
        y = y[: args.n_samples]
    D_np = sklearn_pairwise_distances(X, metric="euclidean").astype(np.float32)

    device = resolve_device(args.device)
    device_setting = str(device)

    all_results: List[Dict[str, Any]] = []

    for exp in config.get("experiments", []):
        exp_name = exp.get("name", "Unnamed Experiment")
        base_params = copy.deepcopy(exp.get("base_params", {}))
        if not base_params:
            raise ValueError(f"Experiment '{exp_name}' lacks 'base_params'.")

        base_params.setdefault("device", device_setting)
        if args.max_iter is not None:
            base_params["max_iter"] = int(args.max_iter)

        searches = exp.get("search", {})
        combos = list(iter_search_points(searches))
        if args.max_configs is not None:
            combos = combos[: args.max_configs]
        if not combos:
            combos = [{}]

        print(f"\n=== Experiment: {exp_name} ({len(combos)} configurations) ===")
        for idx, overrides in enumerate(combos, start=1):
            params = copy.deepcopy(base_params)
            for dotted_key, value in overrides.items():
                set_nested_value(params, dotted_key, value)
            label = f"{exp_name}#{idx}"
            try:
                result = evaluate_config(label, params, D_np, stress_device=args.stress_device)
                result["experiment"] = exp_name
                result["overrides"] = overrides
                all_results.append(result)
                print(
                    f"[{label}] stress={result['stress']:.6f} "
                    f"time={result['fit_time']:.3f}s "
                    f"n_iter={result['n_iter']}"
                )
            except Exception as exc:
                print(f"[{label}] FAILED: {exc}")

    if not all_results:
        print("No successful configurations were evaluated.")
        return

    print("\n=== Top Results Per Experiment ===")
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for res in all_results:
        grouped.setdefault(res["experiment"], []).append(res)

    for exp_name, exp_results in grouped.items():
        top = sorted(exp_results, key=lambda r: r["stress"])[: args.top_k]
        print(f"\nExperiment: {exp_name}")
        for rank, res in enumerate(top, start=1):
            print(
                f"  #{rank}: stress={res['stress']:.6f}, "
                f"time={res['fit_time']:.3f}s, n_iter={res['n_iter']}, "
                f"overrides={res.get('overrides', {})}"
            )

    best_overall = min(all_results, key=lambda r: r["stress"])
    print("\n=== Best Overall Configuration ===")
    print(
        f"{best_overall['model_name']} -> stress={best_overall['stress']:.6f}, "
        f"time={best_overall['fit_time']:.3f}s, n_iter={best_overall['n_iter']}"
    )
    print(f"Overrides: {best_overall.get('overrides', {})}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved raw results to {args.output}")


if __name__ == "__main__":
    main()
