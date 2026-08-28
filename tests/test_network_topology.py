import pytest

from src.network import QuantumNetworkTopology


class TestNetworkTopology:
    """Test network topology."""

    def test_network_topology_validation(self):
        """Test network topology validation."""
        # Mismatched qpu_count and qubits_per_qpu
        with pytest.raises(ValueError):
            QuantumNetworkTopology(
                qpu_count=3,
                qubits_per_qpu=[2, 2],  # Only 2 items, but qpu_count=3
                connectivity=[],
            )

        # Invalid edge references
        with pytest.raises(ValueError):
            QuantumNetworkTopology(
                qpu_count=2,
                qubits_per_qpu=[2, 2],
                connectivity=[(0, 5)],  # QPU 5 doesn't exist
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
