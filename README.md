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