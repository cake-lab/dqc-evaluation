from dataclasses import dataclass, field
from typing import Any

from src.config import PARTITIONING_METRIC_KEYS


@dataclass
class ExperimentResultRow:
    """Structured result row used by the experiment runner and recorder."""

    experiment_details: dict[str, Any]
    status: dict[str, Any] = field(
        default_factory=lambda: {
            "status": "pending",
            "error_message": None,
            "exception_traceback": None,
        }
    )
    input_circuit_metrics: dict[str, Any] = field(default_factory=dict)
    output_circuit_metrics: dict[str, Any] = field(default_factory=dict)
    partitioning_metrics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        benchmark: str,
        network_topology: str,
        num_qpus: int,
        partitioner_name: str,
        circuit_id: str,
    ) -> "ExperimentResultRow":
        return cls(
            experiment_details={
                "benchmark": benchmark,
                "network_topology": network_topology,
                "num_qpus": num_qpus,
                "partitioner": partitioner_name,
                "circuit": circuit_id,
            }
        )

    def mark_skipped(self, reason: str) -> None:
        self.status["status"] = "skipped"
        self.status["error_message"] = reason

    def mark_success(
        self,
        input_circuit_metrics: dict[str, Any],
        output_circuit_metrics: dict[str, Any],
        partitioning_metrics: dict[str, Any],
    ) -> None:
        self.status["status"] = "success"
        self.input_circuit_metrics.update(input_circuit_metrics)
        self.output_circuit_metrics.update(output_circuit_metrics)
        self.partitioning_metrics.update(partitioning_metrics)
        for k in PARTITIONING_METRIC_KEYS:
            self.partitioning_metrics.setdefault(k, None)

    def mark_timeout(
        self, error_message: str, traceback_text: str, timeout_seconds: int
    ) -> None:
        self.status["status"] = f"timeout ({timeout_seconds} seconds)"
        self.status["error_message"] = error_message
        self.status["exception_traceback"] = traceback_text

    def mark_failure(self, error_message: str, traceback_text: str) -> None:
        self.status["status"] = "failed"
        self.status["error_message"] = error_message
        self.status["exception_traceback"] = traceback_text

    def mark_partial_success(
        self, error_message: str, partitioning_metrics: dict[str, Any]
    ) -> None:
        self.status["status"] = "partial_success"
        self.status["error_message"] = error_message
        self.partitioning_metrics.update(partitioning_metrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_details": self.experiment_details,
            "status": self.status,
            "input_circuit_metrics": self.input_circuit_metrics,
            "output_circuit_metrics": self.output_circuit_metrics,
            "partitioning_metrics": self.partitioning_metrics,
        }
