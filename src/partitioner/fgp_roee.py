import contextlib
from time import perf_counter
from typing import ClassVar

from disqco import PartitionedCircuitExtractor, QuantumCircuitHyperGraph
from disqco.parti.fgp.fgp_roee import main_algorithm, set_initial_partition_fgp
from qiskit import QuantumCircuit, transpile

from src.network import QuantumNetworkTopology, topology_to_disqco_network
from src.partitioner.base import Partitioner
from src.partitioner.outcome import (
    PARTITIONER_DISTRIBUTION_COST_METRIC,
    PARTITIONER_PARTITION_EXEC_TIME_METRIC,
    PARTITIONER_POSTPROCESSING_EXEC_TIME_METRIC,
    PARTITIONER_PREPROCESSING_EXEC_TIME_METRIC,
    PARTITIONER_TYPE_METRIC,
    PartitionOutcome,
)


class FGPrOEEPartitioner(Partitioner):
    """
    Partitioner using Fine Grained Partitioning with relaxed Overall Extreme Exchange (FGP-rOEE) (https://arxiv.org/abs/2005.12259) implemented in the DISQCO framework.

    Configuration options:
        - group_gates (bool, default=False)
        - remove_singles (bool, default=False)
        - choose_initial (bool, default=True)
    """

    BASIS_GATES: ClassVar[list[str]] = ["u", "cp"]
    PARTITIONER_TYPE = "FGP-rOEE"

    def __init__(self, **config):
        """
        Initialize FGPrOEEPartitioner.

        Args:
            **config: Algorithm configuration
        """
        super().__init__(**config)
        self.group_gates = config.get("group_gates", False)
        self.remove_singles = config.get("remove_singles", False)
        self.choose_initial = config.get(
            "choose_initial", True
        )  # TODO put to False to enable the rOEE

    def _partition(
        self, circuit: QuantumCircuit, network: QuantumNetworkTopology
    ) -> PartitionOutcome:
        """
        Partition circuit using FGP-rOEE.

        Args:
            circuit: Qiskit circuit
            network: QuantumNetworkTopology

        Returns:
            Partitioned circuit and metrics
        """
        # Convert network
        disqco_network = topology_to_disqco_network(network)

        preprocessing_start_time_s = perf_counter()

        # Transpile to basis gates
        # TODO check what to do with basis gates
        circuit = transpile(circuit, basis_gates=self.BASIS_GATES)

        # Create hypergraph representation
        qc_hypergraph = QuantumCircuitHyperGraph(
            circuit,
            group_gates=self.group_gates,
            qpu_sizes=network.qubits_per_qpu,
        )

        # Initialize partition assignment
        initial_partition = set_initial_partition_fgp(
            network.qubits_per_qpu,
            num_partitions=network.qpu_count,
        )

        preprocessing_exec_time_s = perf_counter() - preprocessing_start_time_s
        self._record_metric(
            PARTITIONER_PREPROCESSING_EXEC_TIME_METRIC,
            preprocessing_exec_time_s,
        )

        partition_start_time_s = perf_counter()

        # Run partitioning
        with contextlib.redirect_stdout(None):  # Mute FGP-rOEE stdout
            partition, distribution_cost, _ = main_algorithm(
                circuit=circuit,
                qpu_info=network.qubits_per_qpu,  # TODO this?
                initial_partition=initial_partition,
                remove_singles=self.remove_singles,
                choose_initial=self.choose_initial,
            )

        partition_exec_time_s = perf_counter() - partition_start_time_s
        self._record_metric(
            PARTITIONER_DISTRIBUTION_COST_METRIC,
            distribution_cost,
        )
        self._record_metric(
            PARTITIONER_PARTITION_EXEC_TIME_METRIC,
            partition_exec_time_s,
        )

        postprocessing_start_time_s = perf_counter()

        # Extract partitioned circuit
        circuit_extractor = PartitionedCircuitExtractor(
            qc_hypergraph,
            disqco_network,
            partition_assignment=partition,
        )
        partitioned_circuit = circuit_extractor.extract_partitioned_circuit()

        postprocessing_exec_time_s = perf_counter() - postprocessing_start_time_s
        self._record_metric(
            PARTITIONER_POSTPROCESSING_EXEC_TIME_METRIC,
            postprocessing_exec_time_s,
        )

        self._record_metric(PARTITIONER_TYPE_METRIC, self.PARTITIONER_TYPE)

        return PartitionOutcome.success(
            partitioned_circuit,
            self._get_metrics(),
        )
