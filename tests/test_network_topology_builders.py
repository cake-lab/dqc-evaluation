"""
Unit tests for quantum circuit partitioners.

Tests cover:
- Partitioner instantiation with unified topologies
- Circuit partitioning produces valid Qiskit circuits
- EPR operations present in output
- Multiple topology types and circuit configurations
"""

import pytest

from src.network import (
    custom_topology,
    fully_connected_topology,
    linear_topology,
    mesh_topology,
)


class TestNetworkTopologyBuilders:
    """Test network topology builder functions."""

    def test_linear_network_topology(self):
        """Test linear network topology creation."""
        network = linear_topology(qpu_count=3, qubits_per_qpu=2)
        assert network.qpu_count == 3
        assert network.total_qubits == 6
        assert len(network.connectivity) == 2  # 3-1 edges
        assert network.coupling_type == "linear"

    def test_mesh_network_topology(self):
        """Test mesh network topology creation."""
        network = mesh_topology(rows=2, cols=2, qubits_per_qpu=1)
        assert network.qpu_count == 4
        assert len(network.connectivity) == 4  # 2x2 mesh has 4 edges
        assert network.coupling_type == "grid"

    def test_fully_connected_network_topology(self):
        """Test fully connected network topology creation."""
        network = fully_connected_topology(qpu_count=3, qubits_per_qpu=2)
        assert network.qpu_count == 3
        assert len(network.connectivity) == 3  # n(n-1)/2 = 3 edges
        assert network.coupling_type == "fully_connected"

    def test_custom_network_topology_with_heterogeneous_qubits(self):
        """Test custom network topology with different qubit counts per QPU."""
        network = custom_topology(
            qpu_count=2, qubits_per_qpu=[3, 2], connectivity=[(0, 1)]
        )
        assert network.qpu_count == 2
        assert network.total_qubits == 5
        assert network.qubits_per_qpu == [3, 2]

    def test_qubit_mapping_generation(self):
        """Test automatic qubit mapping generation."""
        network = linear_topology(qpu_count=3, qubits_per_qpu=2)
        mapping = network.get_qubit_mapping()

        assert len(mapping) == 3
        assert mapping[0] == [0, 1]
        assert mapping[1] == [2, 3]
        assert mapping[2] == [4, 5]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
