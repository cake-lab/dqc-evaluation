import logging
import multiprocessing as mp
import signal
import traceback
from abc import ABC, abstractmethod
from queue import Empty
from typing import Any

from qiskit import QuantumCircuit

from src.network import QuantumNetworkTopology
from src.partitioner.outcome import PARTITIONING_METRIC_KEYS, PartitionOutcome


def _partition_worker(
    partitioner_cls: type,
    config: dict,
    circuit: QuantumCircuit,
    network: QuantumNetworkTopology,
    result_queue,
) -> None:
    """Run partitioning in a child process so native crashes stay contained."""
    try:
        # Create the subclass inside the child process, then run its real partition logic.
        partitioner = partitioner_cls(**config)
        outcome = partitioner._partition(circuit, network)
    except BaseException as exc:  # noqa: BLE001 - report process-level failures
        result_queue.put(
            (
                "exception",
                type(exc).__name__,
                str(exc),
                traceback.format_exc(),
            )
        )
    else:
        result_queue.put(("outcome", outcome))


class Partitioner(ABC):
    """
    Abstract base class for quantum circuit partitioners.
    """

    def __init__(self, **config):
        """
        Initialize partitioner.

        Args:
            **config: Algorithm-specific configuration (varies by subclass)
        """
        self.config = config
        self.run_in_subprocess = bool(config.get("run_in_subprocess", False))
        self._metrics: dict[str, Any] = {}

    def _reset_metrics(self) -> None:
        self._metrics = {}

        for k in PARTITIONING_METRIC_KEYS:
            self._metrics.setdefault(k, None)

    def _record_metric(self, key: str, value: Any) -> None:
        self._metrics[key] = value

    def _record_metrics(self, metrics: dict[str, Any]) -> None:
        self._metrics.update(metrics)

    def _get_metrics(self) -> dict[str, Any]:
        return dict(self._metrics)

    def _partition_in_subprocess(
        self,
        circuit: QuantumCircuit,
        network: QuantumNetworkTopology,
        timeout: int,
    ) -> PartitionOutcome:
        # Spawn keeps crashes isolated and avoids fork warnings in threaded parents.
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue(maxsize=1)
        worker = ctx.Process(
            target=_partition_worker,
            args=(self.__class__, self.config, circuit, network, result_queue),
        )

        worker.start()
        worker.join(timeout)

        if worker.is_alive():
            worker.terminate()
            worker.join()
            timeout_reason = f"partition timed out at {timeout} seconds"
            return PartitionOutcome.timeout(
                timeout_reason,
                metrics=self._get_metrics(),
                traceback=f"TimeoutError: {timeout_reason}",
            )

        try:
            tag, *payload = result_queue.get_nowait()
        except Empty:
            if worker.exitcode == 0:
                return PartitionOutcome.failed(
                    "partition worker exited without returning a result",
                    metrics=self._get_metrics(),
                    traceback="worker exited cleanly but no outcome was produced",
                )

            return PartitionOutcome.failed(
                f"partition worker crashed (exit code {worker.exitcode})",
                metrics=self._get_metrics(),
                traceback=f"worker exit code {worker.exitcode}",
            )

        if tag == "outcome":
            result = payload[0]
            self._metrics = dict(result.metrics)
            return result

        exception_name, exception_message, exception_traceback = payload
        return PartitionOutcome.failed(
            "partition failed inside a worker process",
            metrics=self._get_metrics(),
            traceback=exception_traceback.strip()
            or f"{exception_name}: {exception_message}",
        )

    @abstractmethod
    def _partition(
        self, circuit: QuantumCircuit, network: QuantumNetworkTopology
    ) -> PartitionOutcome:
        """
        Subclass-implemented partition logic.

        Implementations should perform the actual partition and return
        a PartitionOutcome.
        """

    def partition(
        self,
        circuit: QuantumCircuit,
        network: QuantumNetworkTopology,
        timeout: int = 300,
    ) -> PartitionOutcome:
        """
        Public partition method that enforces a timeout and delegates to
        `_partition` implemented by subclasses.

        Args:
            circuit: Qiskit QuantumCircuit
            network: QuantumNetworkTopology
            timeout: timeout in seconds (default 300)

        Returns:
            PartitionOutcome returned by `_partition` or a failed/timeout outcome.
        """
        self._reset_metrics()

        if self.run_in_subprocess:
            return self._partition_in_subprocess(circuit, network, timeout)

        def _handler(signum, frame):
            raise TimeoutError(f"partition timed out at {timeout} seconds")

        old_handler = signal.getsignal(signal.SIGALRM)
        try:
            signal.signal(signal.SIGALRM, _handler)
            signal.alarm(int(timeout))
            result = self._partition(circuit, network)
            if not result.metrics:
                result.metrics.update(self._get_metrics())
            signal.alarm(0)
            return result
        except TimeoutError:
            signal.alarm(0)
            timeout_reason = f"partition timed out at {timeout} seconds"
            return PartitionOutcome.timeout(
                timeout_reason,
                metrics=self._get_metrics(),
                traceback=f"TimeoutError: {timeout_reason}",
            )
        finally:
            signal.alarm(0)
            try:
                signal.signal(signal.SIGALRM, old_handler)
            except (OSError, ValueError) as exc:
                logging.getLogger(__name__).warning(
                    "Failed to restore SIGALRM handler: %s", exc
                )
