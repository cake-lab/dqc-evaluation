# Evaluation of Distributed Quantum Circuit Partitioning Algorithms

[![CI](https://github.com/cake-lab/dqc-evaluation/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/cake-lab/dqc-evaluation/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

A reproducible evaluation framework for benchmarking **distributed quantum circuit partitioning** algorithms under network constraints. It compares partitioning methods across standardized workloads and topologies while extracting circuit-level metrics beyond the usual entanglement cost, enabling a more faithful assessment of their practical performance in distributed quantum execution.

The evaluation methodology and results are described in the paper [*Towards Reproducible Evaluation of Distributed Quantum Circuit Partitioning Algorithms*](https://arxiv.org/abs/2608.27099).

## Repository Structure

```txt
/
├── experiments/      # REPRODUCIBLE EXPERIMENTS
│   ├── data/         # Processed results dataframe in .parquet format
│   ├── notebooks/    # Jupyter notebooks for reproducing paper analyses
│   └── config.yaml   # Configuration file for the experiments
│
├── src/              # SOURCE CODE
│   ├── config/       # Module for loading configurations
│   ├── network/      # Module for network topologies
│   ├── partitioner/  # Module for partitioning algorithms
│   ├── results/      # Module for handling experiment results
│   ├── __init__.py
│   ├── main.py       # Main entry point for running the evaluation
│   └── metrics.py    # Module for computing evaluation metrics
│
├── tests/            # TESTS
├── pyproject.toml    # Python project configuration
├── README.md         # This README file
└── uv.lock
```

## Installation

> [!WARNING]
>
> This project currently depends on [`qceval`](https://github.com/cake-lab/qceval), an internal library that is not yet publicly available. Therefore, the installation instructions below will not work until **`qceval` is released shortly**.

### 0. Prerequisites

- To reproduce the paper results, install the partitioning algorithms by following the instructions for [DISQCO](https://github.com/felix-burt/DISQCO#installation) and [Pytket-DQC](https://github.com/Quantinuum/pytket-dqc#requirements).

- The installation and execution instructions in this README use [uv](https://docs.astral.sh/uv/).

### 1. Clone the repo

```sh
git clone https://github.com/cake-lab/dqc-evaluation.git
cd dqc-evaluation
```

### 2. Install dependencies

```sh
uv sync
```

## Supported Features & Components

The framework supports the following components:

- **[Algorithms](src/partitioner/):**
  - MLFM-R ([DISQCO](https://github.com/felix-burt/DISQCO))
  - FGP-rOEE (implemented in [DISQCO](https://github.com/felix-burt/DISQCO))
  - [Pytket-DQC](https://github.com/Quantinuum/pytket-dqc) (P, PE, ESD variants)
- **[Network Topologies](src/network/):**
  - 2-QPU fully connected
  - 4-QPU linear
  - 4-QPU grid
  - 4-QPU fully connected
- **[Benchmarks](https://github.com/cake-lab/qceval/tree/main/benchmarks):**
  - [QASMBench](https://github.com/pnnl/QASMBench/)
  - Quantum Fourier Transform (QFT)
  - QAOA
  - Quantum Volume (QV)
  - Controlled-Phase (CP) fraction circuits
- **[Evaluation Metrics](https://github.com/cake-lab/qceval/tree/main/src/qceval/metrics):**
  - Partitioning direct costs: e-bit count, execution runtime
  - Standard properties: width, depth, gate count
  - Those defined in the [QASMBench](https://doi.org/10.1145/3550488) and [SupermarQ](https://doi.org/10.1109/HPCA53966.2022.00050) papers, including:
    - Gate density
    - Parallelism
    - Retention lifespan
    - Liveness
    - Entanglement ratio
    - Entanglement variance

## Quick Start

The framework is designed to be run from the command line. The main entry point is `main.py`, which can be executed with the `uv run` command.

The user needs to provide a configuration file in YAML format, which specifies the partitioning algorithm, network topology, benchmark circuits, and evaluation metrics to be used. An example configuration file is provided in [`experiments/config.yaml`](experiments/config.yaml) with explanations for each option. Use the `--config`/`-c` option to specify the path to the configuration file.

<!-- TODO config.yaml explanation -->
<!-- TODO qceval how to use -> .env file -->

To run the framework with the example configuration, use the following command:

```sh
uv run python -m src.main --config tests/config.yaml
```

To run the tests, use the following command:

```sh
uv run pytest tests/
```

## Reproducing Paper Results

For reproducing the experiments and getting the exact results reported in the paper, use the provided configuration file [`experiments/config.yaml`](experiments/config.yaml) and run the following command:

```sh
uv run python -m src.main --config experiments/config.yaml
```

The process is computationally intensive and may take several hours to complete. For that reason, it is recommended to run the split the experiments into smaller batches by modifying the configuration file to specify a subset of the benchmark circuits and network topologies.

Processed datasets are available in the [`experiments/data/results.parquet`](experiments/data/results.parquet) dataframe. Additionally, you can download pre-computed results from [Zenodo (DOI: 10.5281/zenodo.22117678)](https://doi.org/10.5281/zenodo.22117678) or via CLI:

```bash
wget [https://zenodo.org/records/22117678/files/results.tar.gz](https://zenodo.org/records/22117678/files/results.tar.gz)
tar -xzf results.tar.gz -C ./results/
```

The processed results can be used to generate the analysis and plots in the paper without having to run the experiments again. For that, use the provided Jupyter notebooks in the [`experiments/notebooks`](experiments/notebooks) folder.

## How to Extend the Framework

The framework is designed to be modular and extensible. New modules can be added to support additional partitioning algorithms (in [`src/partitioner`](src/partitioner)) and network topologies (in [`src/network`](src/network)).

Extending the available benchmark circuits and evaluation metrics can be done by adding new modules to the [`qceval`](https://github.com/cake-lab/qceval) dependency.

> [!NOTE]
>
> We welcome **contributions from the community** to improve and extend the framework. If you would like to contribute, open an issue or submit a pull request on the GitHub repository.

## Citation

If you use this framework in your research, please cite the following paper:

```
@misc{vela-tamboReproducibleEvaluationDistributed2026,
  title         = {Towards Reproducible Evaluation of Distributed Quantum Circuit Partitioning Algorithms},
  author        = {Vela-Tambo, Javier and Azizov, Davud and Guo, Tian},
  year          = {2026},
  eprint        = {2608.27099},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph}
}
```

## Acknowledgments

This work was supported in part by the National Science Foundation under grant #2426940.
