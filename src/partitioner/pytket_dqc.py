from time import perf_counter

from pytket import Bit, Circuit, OpType
from pytket.circuit import CustomGateDef
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket_dqc.circuits import Distribution
from pytket_dqc.distributors import (
    CoverEmbeddingSteiner,
    CoverEmbeddingSteinerDetached,
    PartitioningAnnealing,
    PartitioningHeterogeneous,
    PartitioningHeterogeneousEmbedding,
)
from pytket_dqc.refiners import (
    BoundaryReallocation,
    DetachedGates,
)
from pytket_dqc.utils import DQCPass
from qiskit import ClassicalRegister, QuantumCircuit

from src.network import QuantumNetworkTopology, topology_to_pytket_dqc_network
from src.partitioner.base import Partitioner
from src.partitioner.outcome import (
    PARTITIONER_DISTRIBUTION_COST_METRIC,
    PARTITIONER_PARTITION_EXEC_TIME_METRIC,
    PARTITIONER_POSTPROCESSING_EXEC_TIME_METRIC,
    PARTITIONER_PREPROCESSING_EXEC_TIME_METRIC,
    PARTITIONER_TYPE_METRIC,
    PartitionOutcome,
)
from src.partitioner.pytket_dqc_safe_allocator import SafeHypergraphPartitioning


class PytketDqcBase(Partitioner):
    """Shared base for pytket-dqc based partitioners.

    Subclasses should override `_get_distributor()` to return an instance
    of a `pytket_dqc.distributors.*` Distributor.
    """

    # Teleportation protocol circuits
    _EPR_MARKER_CIRCUIT: Circuit = None
    _TELEPORT_START_PROTO: Circuit = None
    _TELEPORT_END_PROTO: Circuit = None

    PARTITIONER_TYPE = "Pytket-dqc"

    @classmethod
    def _initialize_teleportation_circuits(cls) -> None:
        """
        Initialize teleportation protocol circuits as class constants;
        """
        if cls._EPR_MARKER_CIRCUIT is not None:
            return

        # Define EPR gate (abstract marker)
        cls._EPR_MARKER_CIRCUIT = Circuit(2)
        epr_gate_def = CustomGateDef.define("EPR", cls._EPR_MARKER_CIRCUIT, [])

        # Protocol for SOURCE QPU: create EPR pair and measure
        cls._TELEPORT_START_PROTO = Circuit(2, 2, name="Teleport_Start")
        cls._TELEPORT_START_PROTO.add_custom_gate(epr_gate_def, [], [0, 1])
        cls._TELEPORT_START_PROTO.CX(0, 1)
        cls._TELEPORT_START_PROTO.H(0)
        cls._TELEPORT_START_PROTO.Measure(0, 0)
        cls._TELEPORT_START_PROTO.Measure(1, 1)

        cls._TELEPORT_END_PROTO = Circuit(1, 2, name="Teleport_End")
        cls._TELEPORT_END_PROTO.X(0, condition_bits=[1], condition_value=1)
        cls._TELEPORT_END_PROTO.Z(0, condition_bits=[0], condition_value=1)

    def __init__(self, **config):
        """
        Initialize PytketDqcPartitioner.

        Args:
            **config: Algorithm configuration
        """
        super().__init__(**config)
        self.seed = config.get("seed", 42)
        self._initialize_teleportation_circuits()

    def _get_distributor(self):
        raise NotImplementedError()

    @staticmethod
    def _strip_barriers(circuit: QuantumCircuit) -> QuantumCircuit:
        """Remove barriers before converting to pytket."""
        stripped_circuit = QuantumCircuit(
            *circuit.qregs, *circuit.cregs, name=circuit.name
        )
        stripped_circuit.global_phase = circuit.global_phase

        for instruction in circuit.data:
            if instruction.operation.name == "barrier":
                continue
            stripped_circuit.append(
                instruction.operation, instruction.qubits, instruction.clbits
            )

        return stripped_circuit

    @staticmethod
    def _strip_terminal_measurements(
        circuit: QuantumCircuit,
    ) -> tuple[QuantumCircuit, list[tuple[int, int]]]:
        """Remove terminal measurements and remember where they were."""
        stripped_circuit = circuit.copy()
        terminal_measurements: list[tuple[int, int]] = []

        while (
            stripped_circuit.data
            and stripped_circuit.data[-1].operation.name == "measure"
        ):
            instruction = stripped_circuit.data.pop()
            qubit_index = circuit.qubits.index(instruction.qubits[0])
            clbit_index = circuit.clbits.index(instruction.clbits[0])
            terminal_measurements.append((qubit_index, clbit_index))

        terminal_measurements.reverse()

        if any(
            instruction.operation.name == "measure"
            for instruction in stripped_circuit.data
        ):
            return stripped_circuit, terminal_measurements

        return stripped_circuit, terminal_measurements

    @staticmethod
    def _has_mid_circuit_measurements(circuit: QuantumCircuit) -> bool:
        """Return True when the circuit contains a measurement before the end."""
        seen_measurement = False
        for instruction in circuit.data:
            if instruction.operation.name == "measure":
                seen_measurement = True
                continue
            if seen_measurement:
                return True
        return False

    @staticmethod
    def _restore_terminal_measurements(
        circuit: QuantumCircuit,
        terminal_measurements: list[tuple[int, int]],
    ) -> QuantumCircuit:
        """Re-attach measurements that were stripped before partitioning."""
        if not terminal_measurements:
            return circuit

        restored_circuit = circuit.copy()
        measurement_register = ClassicalRegister(
            max(clbit_index for _, clbit_index in terminal_measurements) + 1,
            "input_measurements",
        )
        restored_circuit.add_register(measurement_register)

        for qubit_index, clbit_index in terminal_measurements:
            restored_circuit.measure(
                restored_circuit.qubits[qubit_index],
                measurement_register[clbit_index],
            )

        return restored_circuit

    def _partition(
        self, circuit: QuantumCircuit, network: QuantumNetworkTopology
    ) -> PartitionOutcome:
        """
        Partition circuit via pytket-dqc.

        Args:
            circuit: Qiskit circuit
            network: QuantumNetworkTopology

        Returns:
            Partitioned circuit and metrics
        """
        # Convert network
        pytket_dqc_network = topology_to_pytket_dqc_network(network)

        preprocessing_start_time_s = perf_counter()

        circuit = self._strip_barriers(circuit)

        if self._has_mid_circuit_measurements(circuit):
            return PartitionOutcome.skipped(
                "pytket-dqc does not support mid-circuit measurements",
            )

        circuit, terminal_measurements = self._strip_terminal_measurements(circuit)

        # Convert circuit to PyTKet
        pytket_circuit = qiskit_to_tk(circuit)

        # Preprocessing pass
        DQCPass().apply(pytket_circuit)

        preprocessing_exec_time_s = perf_counter() - preprocessing_start_time_s
        self._record_metric(
            PARTITIONER_PREPROCESSING_EXEC_TIME_METRIC,
            preprocessing_exec_time_s,
        )

        partition_start_time_s = perf_counter()

        # Run partitioning
        distribution = self._get_distributor().distribute(
            pytket_circuit,
            pytket_dqc_network,
            seed=self.seed,
        )
        distribution_cost = distribution.cost()

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
        distributed_circuit = distribution.to_pytket_circuit()

        # Implement physical teleportation protocols
        physical_circuit = self._implement_teleportation_protocols(distributed_circuit)

        # Convert back to Qiskit
        partitioned_circuit = tk_to_qiskit(physical_circuit)

        partitioned_circuit = self._restore_terminal_measurements(
            partitioned_circuit,
            terminal_measurements,
        )

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

    def _implement_teleportation_protocols(
        self, distributed_circuit: Circuit
    ) -> Circuit:
        """
        Replace state teleportation markers with explicit gates.

        This implements:
        - starting_process: EPR pair creation + measurement
        - ending_process: Conditional corrections (X, Z gates)

        Args:
            distributed_circuit: PyTKet circuit with teleportation markers

        Returns:
            Circuit with explicit teleportation protocols
        """
        # Use pre-initialized class-level protocols
        start_proto = self._TELEPORT_START_PROTO
        end_proto = self._TELEPORT_END_PROTO

        # Substitute markers with protocols
        new_circuit = Circuit()
        for qubit in distributed_circuit.qubits:
            new_circuit.add_qubit(qubit)
        for bit in distributed_circuit.bits:
            new_circuit.add_bit(bit)

        # Track classical bits for each cut
        classical_bit_queue: list[tuple[Bit, Bit]] = []
        c_bit_counter = 0

        # Process all commands in distributed circuit
        for cmd in distributed_circuit.get_commands():
            if cmd.op.type == OpType.CustomGate:
                gate_name = cmd.op.get_name()

                if gate_name == "starting_process":
                    # Create new classical bits for this cut
                    c0 = Bit("teleport_c", c_bit_counter)
                    c1 = Bit("teleport_c", c_bit_counter + 1)
                    c_bit_counter += 2

                    new_circuit.add_bit(c0)
                    new_circuit.add_bit(c1)
                    classical_bit_queue.append((c0, c1))

                    # Add start protocol
                    new_circuit.add_circuit(start_proto, cmd.args, [c0, c1])

                elif gate_name == "ending_process":
                    # Retrieve classical bits from queue (FIFO)
                    c0, c1 = classical_bit_queue.pop(0)

                    # Add end protocol
                    new_circuit.add_circuit(end_proto, cmd.args, [c0, c1])

                else:
                    # Preserve other custom gates
                    new_circuit.add_gate(cmd.op, cmd.args)
            else:
                # Copy non-custom gates as-is
                new_circuit.add_gate(cmd.op, cmd.args)

        return new_circuit


class SafePartitioningHeterogeneous(PartitioningHeterogeneous):
    """PartitioningHeterogeneous variant that uses the safe allocator."""

    def distribute(self, circ: Circuit, network, **kwargs) -> Distribution:
        distribution = SafeHypergraphPartitioning().allocate(circ, network, **kwargs)
        BoundaryReallocation().refine(distribution, **kwargs)
        return distribution


class SafePartitioningHeterogeneousEmbedding(PartitioningHeterogeneousEmbedding):
    """Embedding-first variant that uses the safe heterogeneous distributor."""

    def distribute(self, circ: Circuit, network, **kwargs) -> Distribution:
        safe_kwargs = dict(kwargs)
        safe_kwargs.setdefault("initial_distributor", SafePartitioningHeterogeneous())
        return super().distribute(circ, network, **safe_kwargs)


class SafeCoverEmbeddingSteinerDetached(CoverEmbeddingSteinerDetached):
    """Detached-gates refinement on top of the safe Steiner embedding."""

    def distribute(self, circ: Circuit, network, **kwargs) -> Distribution:
        safe_kwargs = dict(kwargs)
        safe_kwargs.setdefault("initial_distributor", SafePartitioningHeterogeneous())
        distribution = CoverEmbeddingSteiner().distribute(circ, network, **safe_kwargs)
        DetachedGates().refine(distribution, **kwargs)
        return distribution


class PytketDqcPPartitioner(PytketDqcBase):
    PARTITIONER_TYPE = "Pytket-dqc (P)"

    def _get_distributor(self):
        return PartitioningAnnealing()


class PytketDqcPEPartitioner(PytketDqcBase):
    PARTITIONER_TYPE = "Pytket-dqc (PE)"

    def _get_distributor(self):
        return SafePartitioningHeterogeneousEmbedding()


class PytketDqcESDPartitioner(PytketDqcBase):
    PARTITIONER_TYPE = "Pytket-dqc (ESD)"

    def _get_distributor(self):
        return SafeCoverEmbeddingSteinerDetached()
