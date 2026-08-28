"""Safe Kahypar allocator for pytket-dqc.

Problem: upstream pytket-dqc builds hypergraphs that can contain zero-weight
gate vertices. The Kahypar backend used in this project rejects those graphs,
so PE and ESD runs fail.

What was tried: Kahypar downgrade to 1.3.2 line was not possible in the Python 3.11 environment, and later versions still surfaced the same zero-weight vertex constraint. Downgrading to Python 3.10 was not possible due to the DISQCO dependency.

Solution: preserve the upstream hypergraph partitioning flow, but rewrite the
vertex and block weight calculations so the generated Kahypar inputs stay
valid. The safe allocator keeps the placement semantics intact while ensuring
every vertex and target block weight is positive.
"""

import importlib_resources
import kahypar
from pytket_dqc.allocators.hypergraph_partitioning import HypergraphPartitioning
from pytket_dqc.circuits import HypergraphCircuit
from pytket_dqc.placement import Placement


class SafeHypergraphPartitioning(HypergraphPartitioning):
    """Hypergraph allocator that keeps Kahypar inputs valid.

    pytket-dqc's upstream allocator assigns gate vertices weight 0, but the
    installed Kahypar backend requires every vertex weight to be positive.
    This subclass preserves the allocator flow and only normalizes the inputs.
    """

    @staticmethod
    def _build_server_sizes(network, dist_circ: HypergraphCircuit) -> list[int]:
        server_list = network.get_server_list()
        num_servers = len(server_list)
        qubit_capacity = [len(network.server_qubits[s]) for s in server_list]

        gate_capacity = max(
            len(dist_circ.kahypar_hyperedges()[1])
            - len(dist_circ.get_qubit_vertices()),
            0,
        )
        base_extra, remainder = divmod(gate_capacity, num_servers)

        return [
            qubit_capacity[i] + base_extra + (1 if i < remainder else 0)
            for i in range(num_servers)
        ]

    @staticmethod
    def _build_context(
        num_servers: int,
        server_sizes: list[int],
        ini_path: str,
        seed: int | None,
    ) -> kahypar.Context:
        context = kahypar.Context()
        context.loadINIconfiguration(ini_path)
        context.setK(num_servers)
        context.setCustomTargetBlockWeights(server_sizes)
        context.suppressOutput(True)

        if seed is not None:
            context.setSeed(seed)

        return context

    def initial_distribute(
        self,
        dist_circ: HypergraphCircuit,
        network,
        ini_path: str,
        **kwargs,
    ) -> Placement:
        if not dist_circ.is_valid():
            raise ValueError("This hypergraph is not valid.")

        hyperedge_indices, hyperedges = dist_circ.kahypar_hyperedges()
        num_hyperedges = len(hyperedge_indices) - 1
        num_vertices = len(set(hyperedges))
        num_servers = len(network.get_server_list())
        hyperedge_weights = [1 for _ in range(num_hyperedges)]
        vertex_weights = [1 for _ in range(num_vertices)]
        server_sizes = self._build_server_sizes(network, dist_circ)

        hypergraph = kahypar.Hypergraph(
            num_vertices,
            num_hyperedges,
            hyperedge_indices,
            hyperedges,
            num_servers,
            hyperedge_weights,
            vertex_weights,
        )

        package_path = importlib_resources.files("pytket_dqc")
        default_ini = f"{package_path}/allocators/km1_kKaHyPar_sea20.ini"
        ini_path = kwargs.get("ini_path", default_ini)
        context = self._build_context(
            num_servers,
            server_sizes,
            ini_path,
            kwargs.get("seed", None),
        )

        kahypar.partition(hypergraph, context)

        partition_list = [hypergraph.blockID(i) for i in range(hypergraph.numNodes())]
        placement_dict = {i: server for i, server in enumerate(partition_list)}
        return Placement(placement_dict)
