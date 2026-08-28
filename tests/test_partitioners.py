import time

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT

from src.network import (
    QuantumNetworkTopology,
    linear_topology,
)
from src.partitioner import (
    FGPrOEEPartitioner,
    MLFMRPartitioner,
    Partitioner,
    PytketDqcESDPartitioner,
    PytketDqcPEPartitioner,
    PytketDqcPPartitioner,
)
from src.partitioner.outcome import (
    PARTITIONER_TYPE_METRIC,
    PARTITIONING_METRIC_KEYS,
    PartitionOutcome,
)


class TestMLFMRPartitioner:
    """Test MLFMRPartitioner."""

    def test_instantiation(self):
        """Test MLFMRPartitioner instantiation with linear topology."""
        partitioner = MLFMRPartitioner()
        assert partitioner.group_gates is True
        assert partitioner.passes_per_level == 10

    def test_partitioner_with_custom_config(self):
        partitioner = MLFMRPartitioner(group_gates=False, passes_per_level=20)
        assert partitioner.group_gates is False
        assert partitioner.passes_per_level == 20

    def test_partition_qft_circuit(self):
        network = linear_topology(qpu_count=2, qubits_per_qpu=2)
        partitioner = MLFMRPartitioner()
        circuit = QFT(num_qubits=4)
        decomposed_circuit = circuit.decompose(reps=10)
        outcome = partitioner.partition(decomposed_circuit, network)
        assert outcome.status == "success"
        assert isinstance(outcome.circuit, QuantumCircuit)
        assert outcome.metrics[PARTITIONER_TYPE_METRIC] == "MLFM-R"
        assert set(PARTITIONING_METRIC_KEYS).issubset(outcome.metrics)
        assert outcome.metrics["partition_exec_time_s"] >= 0
        assert outcome.circuit.num_qubits >= 4
        assert len(outcome.circuit.data) > 0


class TestFGPrOEEPartitioner:
    """Test FGPrOEEPartitioner."""

    def test_instantiation(self):
        partitioner = FGPrOEEPartitioner()
        assert partitioner.group_gates is False
        assert partitioner.remove_singles is False
        assert partitioner.choose_initial is True

    def test_partitioner_with_custom_config(self):
        partitioner = FGPrOEEPartitioner(
            group_gates=False, remove_singles=True, choose_initial=False
        )
        assert partitioner.group_gates is False
        assert partitioner.remove_singles is True
        assert partitioner.choose_initial is False

    def test_partition_qft_circuit(self):
        network = linear_topology(qpu_count=2, qubits_per_qpu=2)
        partitioner = FGPrOEEPartitioner()
        circuit = QFT(num_qubits=4)
        decomposed_circuit = circuit.decompose(reps=10)
        outcome = partitioner.partition(decomposed_circuit, network)
        assert outcome.status == "success"
        assert isinstance(outcome.circuit, QuantumCircuit)
        assert outcome.metrics[PARTITIONER_TYPE_METRIC] == "FGP-rOEE"
        assert set(PARTITIONING_METRIC_KEYS).issubset(outcome.metrics)
        assert outcome.metrics["partition_exec_time_s"] >= 0
        assert outcome.circuit.num_qubits >= 4
        assert len(outcome.circuit.data) > 0


class TestPytketDqcVariants:
    """Test the three pytket-dqc variants (P, PE, ESD)."""

    def test_p_variant(self):
        partitioner = PytketDqcPPartitioner()
        assert partitioner.seed == 42
        network = linear_topology(qpu_count=2, qubits_per_qpu=2)
        circuit = QFT(num_qubits=4)
        decomposed_circuit = circuit.decompose(reps=10)
        outcome = partitioner.partition(decomposed_circuit, network)
        assert outcome.status == "success"
        assert isinstance(outcome.circuit, QuantumCircuit)
        assert outcome.metrics[PARTITIONER_TYPE_METRIC] == "Pytket-dqc (P)"
        assert set(PARTITIONING_METRIC_KEYS).issubset(outcome.metrics)
        assert outcome.metrics["partition_exec_time_s"] >= 0
        assert outcome.circuit.num_qubits >= 4
        assert len(outcome.circuit.data) > 0

    def test_p_variant_handles_terminal_measurements(self):
        partitioner = PytketDqcPPartitioner()
        network = linear_topology(qpu_count=2, qubits_per_qpu=2)
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure(0, 0)
        circuit.measure(1, 1)
        decomposed_circuit = circuit.decompose(reps=10)

        outcome = partitioner.partition(decomposed_circuit, network)

        assert outcome.status == "success"
        assert isinstance(outcome.circuit, QuantumCircuit)
        assert outcome.metrics[PARTITIONER_TYPE_METRIC] == "Pytket-dqc (P)"
        assert any(
            instruction.operation.name == "measure"
            for instruction in outcome.circuit.data
        )

    def test_pe_variant(self):
        partitioner = PytketDqcPEPartitioner()
        assert partitioner.seed == 42
        network = linear_topology(qpu_count=2, qubits_per_qpu=2)
        circuit = QFT(num_qubits=4)
        decomposed_circuit = circuit.decompose(reps=10)
        outcome = partitioner.partition(decomposed_circuit, network)
        assert outcome.status == "success"
        assert isinstance(outcome.circuit, QuantumCircuit)
        assert outcome.metrics[PARTITIONER_TYPE_METRIC] == "Pytket-dqc (PE)"
        assert set(PARTITIONING_METRIC_KEYS).issubset(outcome.metrics)
        assert outcome.metrics["partition_exec_time_s"] >= 0
        assert outcome.circuit.num_qubits >= 4
        assert len(outcome.circuit.data) > 0

    def test_esd_variant(self):
        partitioner = PytketDqcESDPartitioner()
        assert partitioner.seed == 42
        network = linear_topology(qpu_count=2, qubits_per_qpu=2)
        circuit = QFT(num_qubits=4)
        decomposed_circuit = circuit.decompose(reps=10)
        outcome = partitioner.partition(decomposed_circuit, network)
        assert outcome.status == "success"
        assert isinstance(outcome.circuit, QuantumCircuit)
        assert outcome.metrics[PARTITIONER_TYPE_METRIC] == "Pytket-dqc (ESD)"
        assert set(PARTITIONING_METRIC_KEYS).issubset(outcome.metrics)
        assert outcome.metrics["partition_exec_time_s"] >= 0
        assert outcome.circuit.num_qubits >= 4
        assert len(outcome.circuit.data) > 0


class TestPartitionerTimeout:
    """Test timeout functionality in partitioners."""

    class SlowPartitioner(Partitioner):
        """Mock partitioner that deliberately sleeps to trigger timeout."""

        def _partition(
            self, circuit: QuantumCircuit, network: QuantumNetworkTopology
        ) -> PartitionOutcome:
            """Sleep for 5 seconds to exceed any reasonable timeout."""
            time.sleep(5)
            return PartitionOutcome.success(
                circuit,
                {PARTITIONER_TYPE_METRIC: "SlowPartitioner"},
            )

    class FastPartitioner(Partitioner):
        """Mock partitioner that completes quickly."""

        def _partition(
            self, circuit: QuantumCircuit, network: QuantumNetworkTopology
        ) -> PartitionOutcome:
            """Sleep briefly then complete."""
            time.sleep(0.1)
            return PartitionOutcome.success(
                circuit,
                {PARTITIONER_TYPE_METRIC: "FastPartitioner"},
            )

    def test_timeout_triggers_error(self):
        """Test that partition returns a timeout outcome with small timeout."""
        partitioner = self.SlowPartitioner()
        network = linear_topology(qpu_count=2, qubits_per_qpu=2)
        circuit = QFT(num_qubits=4)
        decomposed_circuit = circuit.decompose(reps=10)

        # Use 1 second timeout; the partition will sleep 5 seconds
        outcome = partitioner.partition(decomposed_circuit, network, timeout=1)
        assert outcome.status == "timeout"
        assert outcome.reason == "partition timed out at 1 seconds"
        assert outcome.traceback == "TimeoutError: partition timed out at 1 seconds"

    def test_timeout_doesnt_trigger_with_sufficient_time(self):
        """Test that partition completes successfully with sufficient timeout."""

        partitioner = self.FastPartitioner()
        network = linear_topology(qpu_count=2, qubits_per_qpu=2)
        circuit = QFT(num_qubits=4)
        decomposed_circuit = circuit.decompose(reps=10)

        # Use 2 second timeout; the partition only sleeps 0.1 seconds
        outcome = partitioner.partition(decomposed_circuit, network, timeout=2)
        assert outcome.status == "success"
        assert isinstance(outcome.circuit, QuantumCircuit)
        assert outcome.metrics[PARTITIONER_TYPE_METRIC] == "FastPartitioner"


"""
# TODO what to do with these: move to another file at least
class TestPartitionerComparison:
    #Test that all partitioners work with same topology

    def test_all_partitioners_linear_topology(self):
        network = linear_topology(qpu_count=2, qubits_per_qpu=2)
        circuit = QFT(num_qubits=4)
        partitioners = [
            MLFMRPartitioner(),
            FGPrOEEPartitioner(),
            PytketDqcPartitioner(),
        ]
        outputs = []
        for partitioner in partitioners:
            outcome = partitioner.partition(circuit, network)
            assert outcome.status == "success"
            assert isinstance(outcome.circuit, QuantumCircuit)
            assert outcome.metrics["partition_exec_time_s"] >= 0
            assert outcome.circuit.num_qubits >= 4
            outputs.append(outcome.circuit)
        assert len(outputs) == 3

    def test_all_partitioners_mesh_topology(self):
        network = mesh_topology(rows=2, cols=2, qubits_per_qpu=1)
        circuit = QFT(num_qubits=4)
        partitioners = [
            MLFMRPartitioner(),
            FGPrOEEPartitioner(),
            PytketDqcPartitioner(),
        ]
        for partitioner in partitioners:
            partitioned, metrics = partitioner.partition(circuit, network)
            assert isinstance(partitioned, QuantumCircuit)
            assert metrics["partition_exec_time_s"] >= 0
            assert partitioned.num_qubits >= 4
"""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
