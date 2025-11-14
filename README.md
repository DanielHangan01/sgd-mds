# SGD-MDS: A Fast, Modern MDS Implementation

This repository contains a high-performance implementation of **Multidimensional Scaling (MDS)** using **Stochastic Gradient Descent (SGD)** in **PyTorch**.  
The algorithm and its default parameters are based on the paper *“Graph Drawing by Stochastic Gradient Descent”*.

This implementation is designed to be a fast and scalable alternative to classic algorithms like **SMACOF** (used in scikit-learn), especially for large datasets.

---

## Features

- **Scikit-learn Compatible API** - Provides the familiar `.fit()` and `.fit_transform()` methods.
- **GPU Acceleration** - Utilizes PyTorch to run computations on CUDA-enabled GPUs for significant speedups.
- **Advanced Controls** - Includes modern features such as learning rate schedulers and intelligent stopping criteria (e.g., convergence detection).
- **Flexible Benchmarking** - Comes with a benchmarking suite to compare performance and convergence behavior against other models.
- **Configurable Pair Weighting** - Toggle between classic uniform weights and the common \(w_{ij} = 1 / \delta_{ij}\) choice without changing any code.
  Automatic distance flooring keeps inverse-distance weights numerically stable.

---

## Getting Started

This section describes how to set up the environment, install dependencies, and run benchmarks.

### 1. Installation

**Requirements:**  
Python 3.10 or newer.

#### A. Clone the Repository
```bash
git clone https://github.com/DanielHangan01/sgd-mds.git
cd sgd-mds
```

#### B. Create and Activate a Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### C. Install Dependencies

This project uses `pyproject.toml` to manage dependencies.  
Install the project in editable mode (recommended for development):

```bash
pip install -e .
```

Dependencies are specified in `pyproject.toml`:
```toml
[project]
dependencies = [
  "torch>=2.2",
  "numpy>=1.26",
  "scipy>=1.11",
  "scikit-learn>=1.4",
  "matplotlib>=3.7",
  "pyyaml>=6.0",
]
```

---

### 2. Running the Benchmarks

The `benchmarks/` directory contains scripts to evaluate the performance of **SGDMDS** compared to scikit-learn's MDS implementation.

#### A. Data Setup

The benchmarks expect datasets in the `datasets/` directory, with each dataset stored in its own subfolder.

Create the main directory:
```bash
mkdir -p datasets
```

Download the sample datasets using `wget`.

**Seismic Dataset:**
```bash
mkdir -p datasets/seismic
wget -P datasets/seismic/ https://mespadoto.github.io/proj-quant-eval/post/datasets/seismic/X.npy
wget -P datasets/seismic/ https://mespadoto.github.io/proj-quant-eval/post/datasets/seismic/y.npy
```

**Fashion-MNIST Dataset (Subsampled):**
```bash
mkdir -p datasets/fashion_mnist
wget -P datasets/fashion_mnist/ https://mespadoto.github.io/proj-quant-eval/post/datasets/fashion-mnist/X.npy
wget -P datasets/fashion_mnist/ https://mespadoto.github.io/proj-quant-eval/post/datasets/fashion-mnist/y.npy
```

---

#### B. Run a Comparison Benchmark

The script `run_benchmark.py` executes the models defined in `benchmark_config.yaml` and produces a side-by-side plot of the embeddings.

```bash
# Run the benchmark on the seismic dataset
python benchmarks/run_benchmark.py seismic

# Run the benchmark on the Fashion-MNIST dataset
python benchmarks/run_benchmark.py fashion_mnist
```

---

#### C. Run a Convergence Benchmark

The `convergence_benchmark.py` script generates plots showing how the stress value improves over iterations and elapsed time.

```bash
# Run the convergence analysis on the seismic dataset
python benchmarks/convergence_benchmark.py
```

---

### 3. Benchmark Command-Line Options

| Argument | Description | Default |
|-----------|-------------|----------|
| `dataset_name` | (Positional) Name of the dataset folder (e.g. `seismic`) | - |
| `--device` | Device to use (`auto`, `cpu`, `cuda`) | `auto` |
| `--max_iter` | Maximum number of iterations | 300 or 100 |
| `--n_samples` | Subsample the dataset for quick testing | - |
| `--config` | Path to YAML configuration file defining models (`run_benchmark.py`) | `benchmarks/benchmark_config.yaml` |
| `--warmup_runs` | Number of untimed warmup runs before benchmarking (`convergence_benchmark.py`) | - |
| `--stress_weighting` | Weighting used when reporting stress (`uniform` or `inverse_distance`) | `uniform` |
| `--stress_weight_floor_quantile` | Quantile used to clamp inverse-distance stresses (where supported) | `0.01` |

**Examples:**
```bash
# Run a comparison on 1000 Fashion-MNIST samples using GPU
python benchmarks/run_benchmark.py fashion_mnist --n_samples 1000 --device cuda

# Run a convergence analysis for 200 iterations
python benchmarks/convergence_benchmark.py --max_iter 200
```

---

### 4. Customizing Benchmarks with YAML

Benchmark experiments are configured using `benchmarks/benchmark_config.yaml`.

You can modify or extend the experiments by editing this file:

- **Add or remove models:** Edit the `models_to_run` list.  
- **Adjust hyperparameters:** Update the `params` dictionary for any model.  
- **Create new experiments:** Copy the configuration file (e.g., `my_experiment.yaml`) and pass it via the `--config` argument.

### Pair Weighting Options

- The estimator accepts `pair_weighting="inverse_distance"` to enable \(w_{ij}=1 / \delta_{ij}\) both during training and when reporting `stress_`.
- Extremely small distances are automatically floored (1st percentile by default) to avoid runaway updates; override via `pair_weighting_min_delta` or `pair_weighting_floor_quantile`. A `pair_weighting_max_step` cap can be set to limit per-pair displacements (defaults to the paper value of 1.0, i.e., no extra reduction).
- Benchmark scripts expose `--stress_weighting` so every model is evaluated under the same metric.

### Learning-Rate Defaults

`SGDMDS` now uses the paper's recommendations by default (`lr_init="auto"`):

- \( \text{lr}_{\max} = 1 / w_{\min} \) where \(w_{\min}\) is the smallest pair weight.
- \( \text{lr}_{\min} = \varepsilon / w_{\max} \) with \(\varepsilon = \text{paper\_lr\_epsilon}\) (default 0.1).
- For the convergence scheduler, the first phase decays until the step cap stops binding (\(\text{lr} = 1 / w_{\max}\)), after which it switches to the 1/t regime.

Set `lr_init` to a float to override the automatic value or tune `paper_lr_epsilon` for a different final rate.

### Hyperparameter Tuning

`benchmarks/hparam_tuner.py` performs grid-search style sweeps across stoppers, schedulers, and learning-rate/batch-size choices defined in `benchmarks/hparam_search.yaml` (edit this file to change the search space). Example:

```bash
python benchmarks/hparam_tuner.py fashion_mnist \
  --n_samples 2000 \
  --config benchmarks/hparam_search.yaml \
  --top_k 5 \
  --output tuning_results.json
```

Each experiment in the YAML supplies a `base_params` block (fed to `SGDMDS`) plus a `search` section whose dotted keys enumerate the values to try. The tuner prints the best configurations per experiment and optionally saves the full result table as JSON.

---

## Example Workflow

1. Prepare datasets in the `datasets/` directory.  
2. Run a benchmark:
   ```bash
   python benchmarks/run_benchmark.py seismic
   ```
3. View results - embedding plots are saved in the `results/` directory.  
4. Analyze convergence:
   ```bash
   python benchmarks/convergence_benchmark.py
   ```
