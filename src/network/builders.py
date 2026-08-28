from .core import QuantumNetworkTopology


def linear_topology(qpu_count: int, qubits_per_qpu: int) -> QuantumNetworkTopology:
    qubits = [qubits_per_qpu] * qpu_count
    connectivity = [(i, i + 1) for i in range(qpu_count - 1)]
    return QuantumNetworkTopology(
        qpu_count=qpu_count,
        qubits_per_qpu=qubits,
        connectivity=connectivity,
        coupling_type="linear",
    )


def mesh_square_topology(qpu_count: int, qubits_per_qpu: int) -> QuantumNetworkTopology:
    rows = cols = int(qpu_count**0.5)
    if rows * cols != qpu_count:
        raise ValueError(
            f"QPU count {qpu_count} is not a perfect square for mesh topology."
        )
    return mesh_topology(rows, cols, qubits_per_qpu)


def mesh_topology(rows: int, cols: int, qubits_per_qpu: int) -> QuantumNetworkTopology:
    qpu_count = rows * cols
    qubits = [qubits_per_qpu] * qpu_count

    connectivity = []
    for r in range(rows):
        for c in range(cols):
            qpu_id = r * cols + c
            if c + 1 < cols:
                neighbor_id = r * cols + (c + 1)
                connectivity.append((qpu_id, neighbor_id))
            if r + 1 < rows:
                neighbor_id = (r + 1) * cols + c
                connectivity.append((qpu_id, neighbor_id))

    return QuantumNetworkTopology(
        qpu_count=qpu_count,
        qubits_per_qpu=qubits,
        connectivity=connectivity,
        coupling_type="grid",
    )


def fully_connected_topology(
    qpu_count: int, qubits_per_qpu: int
) -> QuantumNetworkTopology:
    qubits = [qubits_per_qpu] * qpu_count
    connectivity = []
    for i in range(qpu_count):
        for j in range(i + 1, qpu_count):
            connectivity.append((i, j))
    return QuantumNetworkTopology(
        qpu_count=qpu_count,
        qubits_per_qpu=qubits,
        connectivity=connectivity,
        coupling_type="fully_connected",
    )


def custom_topology(
    qpu_count: int,
    qubits_per_qpu: list[int],
    connectivity: list[tuple[int, int]],
    qubit_mapping: dict[int, list[int]] | None = None,
) -> QuantumNetworkTopology:
    return QuantumNetworkTopology(
        qpu_count=qpu_count,
        qubits_per_qpu=qubits_per_qpu,
        connectivity=connectivity,
        qubit_mapping=qubit_mapping,
        coupling_type="custom",
    )
