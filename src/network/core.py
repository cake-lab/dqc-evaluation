import json
from dataclasses import dataclass


@dataclass
class QuantumNetworkTopology:
    """
    Representation of a quantum network topology.

    Attributes:
        qpu_count: Number of quantum processing units
        qubits_per_qpu: List of qubit counts for each QPU (index = QPU_id, value = qubit_count)
        connectivity: List of edges (QPU_i, QPU_j) representing connections between QPUs
        coupling_type: Coupling type ('linear', 'grid', 'fully_connected', 'custom').
        qubit_mapping: Optional explicit mapping of physical qubits to QPUs.
                       If None, physical qubits are assigned sequentially by qubits_per_qpu.
    """

    qpu_count: int
    qubits_per_qpu: list[int]
    connectivity: list[tuple[int, int]]
    coupling_type: str = "none"
    qubit_mapping: dict[int, list[int]] | None = None

    def __post_init__(self):
        """
        Validate topology consistency.
        """
        # Validate qpu_count matches qubits_per_qpu length
        if len(self.qubits_per_qpu) != self.qpu_count:
            raise ValueError(
                f"Length of qubits_per_qpu ({len(self.qubits_per_qpu)}) "
                f"must equal qpu_count ({self.qpu_count})"
            )

        # Validate all qubits_per_qpu are positive
        if any(q <= 0 for q in self.qubits_per_qpu):
            raise ValueError("All qubit counts must be positive")

        # Validate connectivity edges reference valid QPUs
        for u, v in self.connectivity:
            if not (0 <= u < self.qpu_count) or not (0 <= v < self.qpu_count):
                raise ValueError(
                    f"Edge ({u}, {v}) references invalid QPU indices. "
                    f"Valid range: [0, {self.qpu_count - 1}]"
                )

        # If qubit_mapping provided, validate consistency
        if self.qubit_mapping is not None:
            total_qubits_mapped = sum(
                len(qubits) for qubits in self.qubit_mapping.values()
            )
            total_qubits_expected = sum(self.qubits_per_qpu)
            if total_qubits_mapped != total_qubits_expected:
                raise ValueError(
                    f"Qubit mapping covers {total_qubits_mapped} qubits, "
                    f"but topology expects {total_qubits_expected} qubits"
                )

    @property
    def total_qubits(self) -> int:
        """Total number of qubits across all QPUs."""
        return sum(self.qubits_per_qpu)

    def get_qubit_mapping(self) -> dict[int, list[int]]:
        """
        Get physical qubit mapping for all QPUs.

        Returns:
            Dict mapping QPU index to list of physical qubit indices
        """
        if self.qubit_mapping is not None:
            return self.qubit_mapping

        # Generate default sequential mapping
        mapping = {}
        qubit_offset = 0
        for qpu_id, qubit_count in enumerate(self.qubits_per_qpu):
            mapping[qpu_id] = list(range(qubit_offset, qubit_offset + qubit_count))
            qubit_offset += qubit_count
        return mapping

    def as_dict(self) -> dict:
        """
        Canonicalize topology as a deterministic dictionary.

        Returns:
            Dictionary with normalized structure for comparison.
        """
        # Normalize edges: ensure each edge is (min, max) and sort them
        normalized_edges = sorted(tuple(sorted([u, v])) for u, v in self.connectivity)

        # Build canonical dict
        canonical = {
            "qpu_count": self.qpu_count,
            "qubits_per_qpu": list(self.qubits_per_qpu),  # Ensure list
            "connectivity": normalized_edges,
        }

        # Include qubit mapping if present, normalized
        if self.qubit_mapping is not None:
            canonical["qubit_mapping"] = {
                k: sorted(self.qubit_mapping[k])
                for k in sorted(self.qubit_mapping.keys())
            }

        # Include coupling type
        canonical["coupling_type"] = self.coupling_type

        return canonical

    def __eq__(self, other: object) -> bool:
        """
        Check equality based on canonical topology structure.

        Two topologies are equal if they have the same QPU count, qubit distribution, and connectivity (regardless of input order).
        """
        if not isinstance(other, QuantumNetworkTopology):
            return False

        return self.as_dict() == other.as_dict()

    def __hash__(self) -> int:
        """
        Hash topology based on canonical structure for use in sets/dicts.
        """
        d = self.as_dict()
        # Convert to JSON for a stable hash
        json_str = json.dumps(d, sort_keys=True)
        return hash(json_str)
