from time import perf_counter
from typing import ClassVar

from disqco import (
    PartitionedCircuitExtractor,
    QuantumCircuitHyperGraph,
    set_initial_partition_assignment,
)
from disqco.graphs.coarsening.coarsener import HypergraphCoarsener
from disqco.parti import FiducciaMattheyses
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


class MLFMRPartitioner(Partitioner):
    """
    Partitioner using Multilevel Partitioning with Fiduccia-Mattheyses heuristic (https://doi.org/10.22331/q-2026-01-22-1984) implemented in the DISQCO framework.

    Configuration options:
        - group_gates (bool, default=True)
        - passes_per_level (int, default=10)
    """

    # TODO add config for the type of coarsening? MLFM-R (recursive), MLFM-W (window), MLFM-B (block)... or only recursive that is the best performing from paper? 10.22331/q-2026-01-22-1984

    PARTITIONER_TYPE = "MLFM-R"
    BASIS_GATES: ClassVar[list[str]] = ["u", "cp"]

    def __init__(self, **config):
        """
        Initialize MLFMRPartitioner.

        Args:
            **config: Algorithm configuration
        """
        super().__init__(**config)
        self.group_gates = config.get("group_gates", True)
        self.passes_per_level = config.get("passes_per_level", 10)

    def _partition(
        self, circuit: QuantumCircuit, network: QuantumNetworkTopology
    ) -> PartitionOutcome:
        """
        Partition circuit using Multilevel Partitioning with Fiduccia-Mattheyses heuristic.

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
        # TODO what to do with basis gates?
        circuit = transpile(circuit, basis_gates=self.BASIS_GATES)

        # Create hypergraph representation
        qc_hypergraph = QuantumCircuitHyperGraph(
            circuit,
            group_gates=self.group_gates,
        )

        # Initialize partition assignment
        initial_assignment = set_initial_partition_assignment(
            qc_hypergraph, disqco_network
        )

        preprocessing_exec_time_s = perf_counter() - preprocessing_start_time_s
        self._record_metric(
            PARTITIONER_PREPROCESSING_EXEC_TIME_METRIC,
            preprocessing_exec_time_s,
        )

        recursive_coarsener = HypergraphCoarsener().coarsen_recursive_batches_mapped
        partitioner = FiducciaMattheyses(
            circuit,
            disqco_network,
            initial_assignment,
            hypergraph=qc_hypergraph,
        )

        partition_start_time_s = perf_counter()

        # Run partitioning
        results = partitioner.multilevel_partition(
            coarsener=recursive_coarsener,
            passes_per_level=self.passes_per_level,
        )
        assignment = results["best_assignment"]
        distribution_cost = results["best_cost"]

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
            partition_assignment=assignment,
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
