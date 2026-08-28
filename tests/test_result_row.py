from src.config import PARTITIONING_METRIC_KEYS
from src.results import ExperimentResultRow


def test_experiment_result_row_success_shape():
    row = ExperimentResultRow.create("bench", "linear", 2, "fgp-roee", "circuit.id")

    row.mark_success(
        input_circuit_metrics={"depth": 10},
        output_circuit_metrics={"depth": 20},
        partitioning_metrics={
            "distribution_cost": 3.5,
            "preprocessing_exec_time_s": 1.0,
            "partition_exec_time_s": 1.5,
            "postprocessing_exec_time_s": 0.5,
            "total_partitioning_exec_time_s": 2.0,
            "total_evaluation_exec_time_s": 0.4,
        },
    )

    payload = row.to_dict()

    assert payload["experiment_details"]["benchmark"] == "bench"
    assert payload["experiment_details"]["network_topology"] == "linear"
    assert payload["status"]["status"] == "success"
    assert payload["input_circuit_metrics"] == {"depth": 10}
    assert payload["output_circuit_metrics"] == {"depth": 20}
    assert set(PARTITIONING_METRIC_KEYS).issubset(payload["partitioning_metrics"])
    assert payload["partitioning_metrics"]["distribution_cost"] == 3.5
    assert payload["partitioning_metrics"]["partition_exec_time_s"] == 1.5
    assert payload["partitioning_metrics"]["total_partitioning_exec_time_s"] == 2.0
    assert payload["partitioning_metrics"]["total_evaluation_exec_time_s"] == 0.4


def test_experiment_result_row_failure_shape():
    row = ExperimentResultRow.create("bench", "linear", 2, "fgp-roee", "circuit.id")

    row.mark_failure("RuntimeError: boom", "Traceback...")

    payload = row.to_dict()

    assert payload["status"]["status"] == "failed"
    assert payload["status"]["error_message"] == "RuntimeError: boom"
    assert payload["status"]["exception_traceback"] == "Traceback..."
