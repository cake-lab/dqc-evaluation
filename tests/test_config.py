import csv
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from qiskit import QuantumCircuit

from src.config import (
    PARTITIONING_METRIC_KEYS,
    RunPaths,
    build_output_namespace,
    build_run_dir_path,
    load_config,
    load_default_config,
)
from src.results import ExperimentRecorder


@pytest.fixture
def cleanup_test_dirs(tmp_path):
    """Fixture to clean up test-created directories after test execution."""
    # Get initial state of directories we might create
    results_path = Path("test_results_config")
    existed_before = results_path.exists()

    yield

    # Only clean up if the test directory didn't exist before and exists now
    if not existed_before and results_path.exists():
        shutil.rmtree(results_path)


def test_load_default_config_contains_expected_sections(cleanup_test_dirs):
    config = load_default_config()

    assert config["experiment"]["benchmarks"] == ["disqco.mfpqc.qft"]
    assert config["experiment"]["network"]["topologies"] == ["linear"]
    assert config["experiment"]["network"]["num_qpus"] == [2, 4]
    assert config["experiment"]["metrics"] is None
    assert config["experiment"]["custom_metrics"] == ["ebit_count"]
    assert config["results"]["output_dir"] == "results"


def test_load_config_builds_run_dir_path(tmp_path, monkeypatch, cleanup_test_dirs):
    """Verify that load_config returns a Config with run_dir set."""
    # Use tmp_path for the HOME to avoid creating 'results' in working directory
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.config.socket.gethostname", lambda: "test-host")

    config = load_config(None)

    assert hasattr(config, "run_paths")
    assert hasattr(config.run_paths, "run_dir")
    assert config.run_paths.run_dir.exists()
    assert config.skip_larger_circuits_after_timeout == 1
    assert config.metrics is None
    assert config.custom_metrics == ["ebit_count"]
    assert config.network_topologies == ["linear"]
    assert config.num_qpus == [2, 4]
    assert "input_circuit_depth" in config.results_columns
    assert "input_ebit_count" in config.results_columns
    assert "output_ebit_count" in config.results_columns
    assert "exception_traceback" not in config.results_columns
    for metric_key in PARTITIONING_METRIC_KEYS:
        assert metric_key in config.results_columns
    assert config.git_commit


def test_finalize_writes_git_commit_file(tmp_path, monkeypatch, cleanup_test_dirs):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.config._get_git_commit_hash", lambda: "abc123")

    config = load_config(None)
    recorder = ExperimentRecorder(
        config.run_paths,
        config.results_columns,
    )

    recorder.write_run_metadata(config)

    assert config.run_paths.config.exists()
    assert config.run_paths.git_commit.exists()
    assert config.run_paths.git_commit.read_text(encoding="utf-8").strip() == "abc123"

    try:
        recorder.finalize(config)
    finally:
        recorder.__exit__(None, None, None)


def test_load_config_deep_merges_user_overrides(tmp_path, cleanup_test_dirs):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
experiment:
  benchmarks:
    - custom.benchmark
  partitioner_configs:
        pytket-dqc-p:
            seed: 7
results:
    output_dir: test_results_config
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.benchmarks == ["custom.benchmark"]
    assert config.partitioner_configs["pytket-dqc-p"]["seed"] == 7
    assert config.network_topologies == ["linear"]
    assert config.num_qpus == [2, 4]
    assert config.custom_metrics == ["ebit_count"]
    assert str(config.run_paths.run_dir).startswith("test_results_config")


def test_load_config_respects_explicit_metric_list(tmp_path, cleanup_test_dirs):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
experiment:
    metrics:
        - circuit_depth
results:
    output_dir: test_results_config
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.metrics == ["circuit_depth"]
    assert config.custom_metrics == ["ebit_count"]
    assert "input_circuit_depth" in config.results_columns
    assert "output_circuit_depth" in config.results_columns
    assert "input_ebit_count" in config.results_columns
    assert "input_qubit_count" not in config.results_columns


def test_build_run_dir_path_creates_timestamped_file(
    tmp_path, monkeypatch, cleanup_test_dirs
):
    monkeypatch.setattr("src.config.socket.gethostname", lambda: "test-host")
    results_path = build_run_dir_path(
        {
            "output_dir": tmp_path / "runs",
        },
        now_fn=lambda: datetime(2026, 5, 13, 12, 34, 56, tzinfo=UTC),
    )

    assert results_path == Path(tmp_path / "runs" / "2026-05-13_12-34-56_test-host")
    assert results_path.exists()


def test_build_run_dir_path_includes_name_and_hostname(
    tmp_path, monkeypatch, cleanup_test_dirs
):
    monkeypatch.setattr("src.config.socket.gethostname", lambda: "test-host")
    results_path = build_run_dir_path(
        {
            "output_dir": tmp_path / "runs",
            "name": "my experiment",
        },
        now_fn=lambda: datetime(2026, 5, 13, 12, 34, 56, tzinfo=UTC),
    )

    assert results_path == Path(
        tmp_path / "runs" / "2026-05-13_12-34-56_test-host_my-experiment"
    )
    assert results_path.exists()


def test_build_output_namespace_is_short_and_stable():
    namespace = build_output_namespace(
        {"partitioner": "FGP-rOEE", "network_topology": "linear", "num_qpus": 4},
    )

    assert (
        namespace
        == Path("partitioner-fgp-roee") / "network-topology-linear" / "num-qpus-4"
    )


def test_experiment_recorder_separates_qasm_by_run_namespace(tmp_path):
    run_dir = tmp_path / "results" / "2026-05-21_12-00-00"
    run_paths = RunPaths(
        run_dir=run_dir,
        output_root=run_dir / "output",
        csv=run_dir / "results.csv",
        parquet=run_dir / "results.parquet",
        summary=run_dir / "summary.md",
        config=run_dir / "config.yaml",
        git_commit=run_dir / "git_commit.txt",
    )

    recorder = ExperimentRecorder(
        run_paths,
        ["partitioner", "network_topology", "num_qpus", "circuit", "status"],
    )
    circuit = QuantumCircuit(1)
    circuit.h(0)

    recorder.record(
        {
            "experiment_details": {
                "partitioner": "mlfm-r",
                "network_topology": "linear",
                "num_qpus": 2,
                "circuit": "disqco.mfpqc.qft.n16.qft_n16",
            },
            "status": {
                "status": "success",
                "error_message": None,
            },
            "input_circuit_metrics": {},
            "output_circuit_metrics": {},
            "partitioning_metrics": {
                "distribution_cost": 7,
                "preprocessing_exec_time_s": 0.1,
                "partition_exec_time_s": 0.2,
                "postprocessing_exec_time_s": 0.3,
                "total_partitioning_exec_time_s": 0.4,
                "total_evaluation_exec_time_s": 0.5,
            },
        },
        partitioned_circuit=circuit,
    )
    recorder.record(
        {
            "experiment_details": {
                "partitioner": "pytket-dqc-p",
                "network_topology": "linear",
                "num_qpus": 4,
                "circuit": "disqco.mfpqc.qft.n16.qft_n16",
            },
            "status": {
                "status": "success",
                "error_message": None,
            },
            "input_circuit_metrics": {},
            "output_circuit_metrics": {},
            "partitioning_metrics": {
                "distribution_cost": 9,
                "preprocessing_exec_time_s": 0.1,
                "partition_exec_time_s": 0.2,
                "postprocessing_exec_time_s": 0.3,
                "total_partitioning_exec_time_s": 0.4,
                "total_evaluation_exec_time_s": 0.5,
            },
        },
        partitioned_circuit=circuit,
    )

    first_path = (
        run_dir
        / "output"
        / "partitioner-mlfm-r"
        / "network-topology-linear"
        / "num-qpus-2"
        / "disqco"
        / "mfpqc"
        / "qft"
        / "n16"
        / "qft_n16_partitioned.qasm"
    )
    second_path = (
        run_dir
        / "output"
        / "partitioner-pytket-dqc-p"
        / "network-topology-linear"
        / "num-qpus-4"
        / "disqco"
        / "mfpqc"
        / "qft"
        / "n16"
        / "qft_n16_partitioned.qasm"
    )
    first_json = first_path.with_name("qft_n16_result.json")
    second_json = second_path.with_name("qft_n16_result.json")

    assert first_json.exists()
    assert second_json.exists()
    assert first_json != second_json

    # Verify JSON contains status field
    with open(first_json, "r", encoding="utf-8") as fh:
        obj = json.load(fh)
    assert "status" in obj
    assert "partitioning_metrics" in obj


def test_result_json_groups_fields_while_csv_stays_flat(tmp_path):
    run_dir = tmp_path / "results" / "2026-05-21_12-30-00"
    run_paths = RunPaths(
        run_dir=run_dir,
        output_root=run_dir / "output",
        csv=run_dir / "results.csv",
        parquet=run_dir / "results.parquet",
        summary=run_dir / "summary.md",
        config=run_dir / "config.yaml",
        git_commit=run_dir / "git_commit.txt",
    )

    recorder = ExperimentRecorder(
        run_paths,
        [
            "benchmark",
            "network_topology",
            "num_qpus",
            "partitioner",
            "circuit",
            "status",
            "error_message",
            "input_circuit_depth",
            "output_circuit_depth",
            "total_partitioning_exec_time_s",
            "total_evaluation_exec_time_s",
        ],
    )

    row = {
        "experiment_details": {
            "benchmark": "disqco.mfpqc.qft",
            "network_topology": "linear",
            "num_qpus": 2,
            "partitioner": "fgp-roee",
            "circuit": "disqco.mfpqc.qft.n16.qft_n16",
        },
        "status": {
            "status": "failed",
            "error_message": "RuntimeError: boom",
        },
        "input_circuit_metrics": {"circuit_depth": 31},
        "output_circuit_metrics": {"circuit_depth": None},
        "partitioning_metrics": {
            "distribution_cost": None,
            "preprocessing_exec_time_s": None,
            "partition_exec_time_s": None,
            "postprocessing_exec_time_s": None,
            "total_partitioning_exec_time_s": None,
            "total_evaluation_exec_time_s": None,
        },
    }

    recorder.record(row, partitioned_circuit=None)

    json_path = (
        run_dir
        / "output"
        / "partitioner-fgp-roee"
        / "network-topology-linear"
        / "num-qpus-2"
        / "disqco"
        / "mfpqc"
        / "qft"
        / "n16"
        / "qft_n16_result.json"
    )
    assert json_path.exists()

    with open(json_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    assert set(payload.keys()) == {
        "experiment_details",
        "status",
        "input_circuit_metrics",
        "output_circuit_metrics",
        "partitioning_metrics",
    }
    assert payload["experiment_details"] == {
        "benchmark": "disqco.mfpqc.qft",
        "network_topology": "linear",
        "num_qpus": 2,
        "partitioner": "fgp-roee",
        "circuit": "disqco.mfpqc.qft.n16.qft_n16",
    }
    assert payload["status"]["status"] == "failed"
    assert payload["status"]["error_message"] == "RuntimeError: boom"
    assert payload["input_circuit_metrics"] == {"circuit_depth": 31}
    assert payload["output_circuit_metrics"] == {"circuit_depth": None}
    assert payload["partitioning_metrics"] == {
        "distribution_cost": None,
        "preprocessing_exec_time_s": None,
        "partition_exec_time_s": None,
        "postprocessing_exec_time_s": None,
        "total_partitioning_exec_time_s": None,
        "total_evaluation_exec_time_s": None,
    }

    with open(run_paths.csv, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        csv_row = next(reader)

    assert csv_row["benchmark"] == "disqco.mfpqc.qft"
    assert csv_row["network_topology"] == "linear"
    assert csv_row["input_circuit_depth"] == "31"
    assert "exception_traceback" not in csv_row
