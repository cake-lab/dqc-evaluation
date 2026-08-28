from disqco import QuantumNetwork
from pytket_dqc.networks import NISQNetwork

from .core import QuantumNetworkTopology


def topology_to_disqco_network(
    topology: QuantumNetworkTopology,
) -> QuantumNetwork:
    """
    Create a DISQCO QuantumNetwork from a QuantumNetworkTopology.

    Args:
        topology: QuantumNetworkTopology instance

    Returns:
        DISQCO QuantumNetwork instance configured with the topology
    """

    network = QuantumNetwork(
        qpu_sizes=topology.qubits_per_qpu,
        qpu_connectivity=topology.connectivity,
    )
    return network


def topology_to_pytket_dqc_network(
    topology: QuantumNetworkTopology,
) -> NISQNetwork:
    """
    Create a Pytket-dqc NISQNetwork from a QuantumNetworkTopology.

    Args:
        topology: QuantumNetworkTopology instance

    Returns:
        Pytket-dqc NISQNetwork instance configured with the topology
    """

    # Convert edges to list format
    server_coupling = [list(edge) for edge in topology.connectivity]

    # Get physical qubit mapping
    server_qubits = topology.get_qubit_mapping()

    network = NISQNetwork(
        server_coupling=server_coupling,
        server_qubits=server_qubits,
    )
    return network
