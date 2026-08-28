import re
import socket
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from src.network import NETWORK_TOPOLOGY_BUILDERS
from src.partitioner import PARTITIONERS
from src.partitioner.outcome import PARTITIONING_METRIC_KEYS

EXPERIMENT_METRIC_KEYS = (
    "total_partitioning_exec_time_s",
    "total_evaluation_exec_time_s",
)
PARTITIONING_METRIC_KEYS = (
    *PARTITIONING_METRIC_KEYS,
    *EXPERIMENT_METRIC_KEYS,
)
DEFAULT_CONFIG_FILE = "config.yaml"
RUN_TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"
OUTPUT_NAMESPACE_KEYS = ["partitioner", "network_topology", "num_qpus"]


@dataclass
class RunPaths:
    """Encapsulates all file paths for an experiment run."""

    run_dir: Path
    output_root: Path
    csv: Path
    parquet: Path
    summary: Path
    config: Path
    git_commit: Path

    def __post_init__(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Configuration for the experiment"""

    benchmarks: list[str]
    network_topologies: list[str]
    num_qpus: list[int]
    partitioner_ids: list[str]
    partitioners: list[tuple[str, type[Any]]]
    partitioner_configs: dict
    metrics: list[str] | None
    custom_metrics: list[str]
    exec_time_columns: list[str]
    timeout_seconds: int
    skip_larger_circuits_after_timeout: int
    run_paths: RunPaths
    results_columns: list[str]
    git_commit: str
    name: str | None
    description: str | None


def _load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise TypeError(f"Configuration file {path} must contain a YAML mapping")

    return loaded


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)

    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)

    return merged


def _resolve_partitioners(partitioner_ids: list[str]) -> list[tuple[str, type[Any]]]:
    resolved: list[tuple[str, type[Any]]] = []

    for pid in partitioner_ids:
        key = str(pid).lower().replace("_", "-")
        if key not in PARTITIONERS:
            available = ", ".join(sorted(PARTITIONERS.keys()))
            raise KeyError(f"Unknown partitioner '{pid}'. Available: {available}")
        resolved.append((pid, PARTITIONERS[key]))

    return resolved


def _resolve_network_topologies(topology_names: list[str]) -> list[str]:
    resolved: list[str] = []

    for topology_name in topology_names:
        if topology_name not in NETWORK_TOPOLOGY_BUILDERS:
            available = ", ".join(sorted(NETWORK_TOPOLOGY_BUILDERS.keys()))
            raise KeyError(
                f"Unknown network topology '{topology_name}'. Available: {available}"
            )
        resolved.append(topology_name)

    return resolved


def _normalize_timeout_skip_count(value: Any) -> int:
    """Normalize the timeout-skip setting to a non-negative integer."""
    if value is None:
        return 1

    if isinstance(value, int):
        if isinstance(value, bool):
            raise TypeError(
                "skip_larger_circuits_after_timeout must be a non-negative integer"
            )
        if value < 0:
            raise ValueError(
                "skip_larger_circuits_after_timeout must be greater than or equal to 0"
            )
        return value

    raise TypeError(
        "skip_larger_circuits_after_timeout must be a boolean or a non-negative integer"
    )


def _builtin_metric_ids() -> list[str]:
    """Return the built-in qceval metric ids."""
    from qceval import CircuitMetricsEvaluator

    evaluator = CircuitMetricsEvaluator()
    return list(evaluator.metric_ids)


def build_metric_ids(metrics: list[str] | None, custom_metrics: list[str]) -> list[str]:
    """Build the ordered metric ids used for evaluation and result columns."""
    effective_metrics = _builtin_metric_ids() if metrics is None else list(metrics)
    return effective_metrics + list(custom_metrics)


def _build_results_columns(
    metrics: list[str] | None,
    custom_metrics: list[str],
    exec_time_columns: list[str],
) -> list[str]:
    base_columns = [
        "benchmark",
        "network_topology",
        "num_qpus",
        "partitioner",
        "circuit",
        "status",
        "error_message",
    ]
    effective_metrics = build_metric_ids(metrics, custom_metrics)
    input_circuit_metric_columns = [f"input_{m}" for m in effective_metrics]
    output_circuit_metric_columns = [f"output_{m}" for m in effective_metrics]
    partitioning_metrics_columns = list(PARTITIONING_METRIC_KEYS)
    return (
        base_columns
        + input_circuit_metric_columns
        + output_circuit_metric_columns
        + partitioning_metrics_columns
    )


def _slugify_namespace_token(value: Any) -> str:
    """Convert a namespace token into a stable filesystem-friendly fragment."""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "unknown"


def build_output_namespace(row: dict[str, Any]) -> Path:
    """Build an ordered filesystem namespace for a row of experiment results."""
    parts: list[str] = []
    source_row = (
        row.get("experiment_details")
        if isinstance(row.get("experiment_details"), dict)
        else row
    )

    for key in OUTPUT_NAMESPACE_KEYS:
        value = source_row.get(key)
        if value is None:
            continue
        parts.append(
            f"{_slugify_namespace_token(key)}-{_slugify_namespace_token(value)}"
        )

    return Path(*parts) if parts else Path("default")


def build_run_dir_path(results_config: dict[str, Any], now_fn: Any = None) -> Path:
    """Build and return a timestamped run directory for an experiment."""
    if now_fn is None:
        from datetime import datetime

        now_fn = datetime.now

    output_dir = Path(results_config["output_dir"])
    timestamp = now_fn().strftime(RUN_TIMESTAMP_FORMAT)
    hostname = _slugify_namespace_token(socket.gethostname())
    # If a name is provided, append a slugified suffix after the timestamp
    name = results_config.get("name") or ""
    if isinstance(name, str) and name.strip():
        suffix = _slugify_namespace_token(name)
        run_dir = output_dir / f"{timestamp}_{hostname}_{suffix}"
    else:
        run_dir = output_dir / f"{timestamp}_{hostname}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _get_git_commit_hash() -> str:
    """Return the current git commit hash if the workspace is a git repository."""
    project_root = Path(__file__).resolve().parents[2]

    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"

    return completed.stdout.strip() or "unknown"


def load_config(config_path: str | Path | None = None) -> Config:
    """Load and merge the default config with an optional user config path."""
    default_path = resources.files(__name__).joinpath(DEFAULT_CONFIG_FILE)
    cfg_map = _load_yaml_file(Path(default_path))

    if config_path is not None:
        user_cfg = _load_yaml_file(Path(config_path))
        cfg_map = _deep_merge(cfg_map, user_cfg)

    experiment = cfg_map["experiment"]
    results = cfg_map["results"]

    network_config = experiment.get("network", {})
    if not isinstance(network_config, dict):
        raise TypeError("experiment.network must be a YAML mapping")

    exec_time_columns = experiment.get("exec_time_columns", [])
    metrics = experiment.get("metrics")
    custom_metrics = experiment.get("custom_metrics", [])

    run_dir = build_run_dir_path(results)

    resolved = _resolve_partitioners(experiment.get("partitioners", []))
    network_topologies = _resolve_network_topologies(
        network_config.get("topologies", ["linear"])
    )

    results_columns = _build_results_columns(metrics, custom_metrics, exec_time_columns)

    run_paths = RunPaths(
        run_dir=run_dir,
        output_root=run_dir / "output",
        csv=run_dir / "results.csv",
        parquet=run_dir / "results.parquet",
        summary=run_dir / "summary.md",
        config=run_dir / "config.yaml",
        git_commit=run_dir / "git_commit.txt",
    )

    config = Config(
        benchmarks=experiment.get("benchmarks", []),
        network_topologies=network_topologies,
        num_qpus=network_config.get("num_qpus", []),
        partitioner_ids=experiment.get("partitioners", []),
        partitioners=resolved,
        partitioner_configs=experiment.get("partitioner_configs", {}),
        metrics=metrics,
        custom_metrics=custom_metrics,
        exec_time_columns=exec_time_columns,
        timeout_seconds=experiment.get("timeout_seconds", 300),
        skip_larger_circuits_after_timeout=_normalize_timeout_skip_count(
            experiment.get("skip_larger_circuits_after_timeout", 1)
        ),
        run_paths=run_paths,
        results_columns=results_columns,
        git_commit=_get_git_commit_hash(),
        name=results.get("name"),
        description=results.get("description"),
    )

    config.raw_config = cfg_map

    return config


def load_default_config() -> dict[str, Any]:
    """Return the default configuration mapping from the bundled YAML file."""
    default_path = resources.files(__name__).joinpath(DEFAULT_CONFIG_FILE)
    return _load_yaml_file(Path(default_path))
