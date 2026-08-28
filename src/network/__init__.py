from .adapters import (
    topology_to_disqco_network,
    topology_to_pytket_dqc_network,
)
from .builders import (
    custom_topology,
    fully_connected_topology,
    linear_topology,
    mesh_square_topology,
    mesh_topology,
)
from .core import QuantumNetworkTopology

__all__ = [
    "QuantumNetworkTopology",
    "custom_topology",
    "fully_connected_topology",
    "linear_topology",
    "mesh_square_topology",
    "mesh_topology",
    "topology_to_disqco_network",
    "topology_to_pytket_dqc_network",
]


NETWORK_TOPOLOGY_BUILDERS = {
    "linear": linear_topology,
    "mesh": mesh_square_topology,
    "fully_connected": fully_connected_topology,
}


def build_network_topology(
    topology_name: str, qpu_count: int, qubits_per_qpu: int
) -> QuantumNetworkTopology:
    """Build a network topology from its canonical name."""
    if topology_name not in NETWORK_TOPOLOGY_BUILDERS:
        available = ", ".join(sorted(NETWORK_TOPOLOGY_BUILDERS.keys()))
        raise KeyError(
            f"Unknown network topology '{topology_name}'. Available: {available}"
        )

    return NETWORK_TOPOLOGY_BUILDERS[topology_name](qpu_count, qubits_per_qpu)
