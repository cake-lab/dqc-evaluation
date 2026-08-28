from dataclasses import dataclass, field
from typing import Any

from qiskit import QuantumCircuit

PARTITIONER_DISTRIBUTION_COST_METRIC = "distribution_cost"
PARTITIONER_PREPROCESSING_EXEC_TIME_METRIC = "preprocessing_exec_time_s"
PARTITIONER_PARTITION_EXEC_TIME_METRIC = "partition_exec_time_s"
PARTITIONER_POSTPROCESSING_EXEC_TIME_METRIC = "postprocessing_exec_time_s"
PARTITIONER_TYPE_METRIC = "partitioner_type"

PARTITIONING_METRIC_KEYS = (
    PARTITIONER_DISTRIBUTION_COST_METRIC,
    PARTITIONER_TYPE_METRIC,
    PARTITIONER_PREPROCESSING_EXEC_TIME_METRIC,
    PARTITIONER_PARTITION_EXEC_TIME_METRIC,
    PARTITIONER_POSTPROCESSING_EXEC_TIME_METRIC,
)


@dataclass(slots=True)
class PartitionOutcome:
    """Structured result returned by partitioners."""

    status: str
    circuit: QuantumCircuit | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    traceback: str | None = None

    @classmethod
    def success(
        cls,
        circuit: QuantumCircuit,
        metrics: dict[str, Any],
    ) -> "PartitionOutcome":
        return cls(
            status="success",
            circuit=circuit,
            metrics=dict(metrics),
        )

    @classmethod
    def skipped(
        cls,
        reason: str,
        metrics: dict[str, Any] | None = None,
    ) -> "PartitionOutcome":
        return cls(
            status="skipped",
            metrics=dict(metrics or {}),
            reason=reason,
        )

    @classmethod
    def timeout(
        cls,
        reason: str,
        circuit: QuantumCircuit | None = None,
        metrics: dict[str, Any] | None = None,
        traceback: str | None = None,
    ) -> "PartitionOutcome":
        return cls(
            status="timeout",
            circuit=circuit,
            metrics=dict(metrics or {}),
            reason=reason,
            traceback=traceback,
        )

    @classmethod
    def failed(
        cls,
        reason: str,
        circuit: QuantumCircuit | None = None,
        metrics: dict[str, Any] | None = None,
        traceback: str | None = None,
    ) -> "PartitionOutcome":
        return cls(
            status="failed",
            circuit=circuit,
            metrics=dict(metrics or {}),
            reason=reason,
            traceback=traceback,
        )
